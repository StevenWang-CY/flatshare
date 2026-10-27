#!/usr/bin/env python3
"""host.py: one host that is both the GPU node and the control host (single-host mode, docs/HARDWARE.md).

    host.py env NODE                  write NODE/env.sh, which every command on this host sources (the supervisor)
    host.py units NODE                link the node's units and cyan's into the user manager, with this host's drop-ins
    host.py config NODE               write the control plane's configuration, keeping what exists
    host.py jit-check NODE [--arch X.Y]   build FlashInfer's sampling module as the engine's first start would
    host.py preflight NODE [--gpu] [--out FILE]   check the host; write NODE/preflight.json
    host.py warm NODE [--configs FILE]    start the engine once per compile configuration; write cache/warm.json
    host.py with-engine-env NODE CMD...   run CMD with the engine unit's environment (profile.py check-vllm)

NODE is the node's directory as deploy/a100/bootstrap.sh makes it: profile.py's rendering (node.json, the engine and
STT configs, the GPU units) and host.json, the host's facts (profile.py detect).

`env` writes NODE/env.sh: nothing where systemd's user manager, outliving the session (lingering), runs the units; else
deploy/bin/flatshare-svc runs them, and env.sh puts its systemctl and journalctl shims first on PATH and gives
XDG_RUNTIME_DIR flatshare-svc's own directory where the session has none. FLATSHARE_SUPERVISOR=flatshare-svc makes
flatshare-svc run them where systemd could (a host whose kernel refuses the voice plane's namespace, with no sudo to
allow it: the voice plane then runs in the host's namespace, degraded, where under systemd the preflight refuses), and
env.sh keeps the choice; FLATSHARE_SUPERVISOR=auto (or unset) lets the host's facts decide. `units` links the node's GPU
units and cyan's control-plane units (deploy/cyan/systemd: the gateway and its socket unit, LiveKit, the voice-plane
namespace and the voice worker; no tunnel, flatshare-tunnel@, since the GPU servers are this host's) into
~/.config/systemd/user, with drop-ins it writes into NODE/single-host: the cores of the node's core plan for the gateway
and the voice plane; the voice worker's STT and TTS on this host's own sockets ($XDG_RUNTIME_DIR/flatshare/stt.sock and
tts.sock, where cyan has the tunnel's phastform-stt.sock and -tts.sock); the voice plane in the host's network namespace
where unprivileged user namespaces are refused; and the vad-turns worker variant, which E2 points need. `config` writes
~/.config/flatshare: gateway.toml from deploy/cyan/gateway.example.toml with the upstream on the engine's own socket,
the voice plane's socket under this host's runtime directory and the digests of new tenant keys (kept in
gateway-keys.env), livekit.env, the cache-salt secret, the metrics key, and e2-host.toml, which points flatshare-bench
e2 run at the engine's and the gateway's sockets, the core plan's load-generator cores, the node's profile and its
preflight. Secrets are written once, mode 600, and never replaced.

`jit-check` builds FlashInfer's top-k/top-p sampler, the kernel the engine JIT-compiles at its first start on this
card (vLLM 0.32.0 samples with FlashInfer on CUDA cards of more than 16 SMs), with the engine unit's environment and
the GPU's architecture, into the engine's own FlashInfer workspace: a check of the compiler, its headers and the link
that needs no GPU, and so runs before the billed clock. `preflight` checks what the campaign depends on
(docs/HARDWARE.md, the preflight): each check ok, degraded with a named reason and what that changes in what results
mean, refused, or pending (it needs the GPU or the engine's first start); with --gpu it also runs the checks that
need the card. `warm` starts
the GPU units once per compile configuration the campaign uses (vLLM's compile cache is keyed by the step cap, not
by the scheduler class, so make-room shares the frozen engine's), each to ready, and keeps each start's time, the
engine log's kernel, attention and cache lines, /server_info's scheduler and /metrics' cache geometry, with the
measurement drop-in on (variant.sh measure on: /server_info is a development route).

Standard library only, so that it runs before any venv exists.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import ipaddress
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent
CYAN = DEPLOY / "cyan"
GIB = 1 << 30


def _profile_module():
    """deploy/profile.py, once (the tests load it by the same name)."""
    if module := sys.modules.get("flatshare_profile"):
        return module
    spec = importlib.util.spec_from_file_location("flatshare_profile", DEPLOY / "profile.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["flatshare_profile"] = module
    spec.loader.exec_module(module)
    return module


profile = _profile_module()

CONTROL = (
    "flatshare-gateway.socket",
    "flatshare-gateway.service",
    "flatshare-livekit.service",
    "flatshare-voicenet.service",
    "flatshare-voice-worker.service",
)
"""cyan's units that a single host runs beside the GPU node's: not SMG (E2 arm 8b's), nor a tunnel (flatshare-tunnel@),
since the control plane reaches this host's GPU servers on their own sockets."""
WORKER_VARIANTS = ("vad-turns",)
"""cyan's worker variants linked for good: every E2 point runs with fixed endpointing (ops/RUNBOOK.md)."""
SUPERVISORS = ("auto", "flatshare-svc", "systemd")
"""FLATSHARE_SUPERVISOR's values: auto, the host's facts decide; flatshare-svc, deploy/bin/flatshare-svc runs the units
where systemd could. env.sh records the supervisor a stage chose, flatshare-svc or systemd, which the later stages keep
(bootstrap.sh)."""
PRIVATE_SERVERS = ("flatshare-engine.service", "flatshare-stt.service", "flatshare-tts.service")
"""The GPU node's servers, each of which deploy/bin/in-private-net runs in a network namespace of its own."""
UNITS = ("flatshare.target", *PRIVATE_SERVERS, *CONTROL)
"""Every unit of this host's: the GPU node's and the control plane's."""
GATEWAY_URL = "http://127.0.0.1:18080"
"""The gateway's TCP listener, AIPerf's way in (gateway.toml's first [[listen]])."""
WARM = (
    ("frozen", "frozen", ()),
    (
        "2048-0",
        "frozen",
        ("--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "0"),
    ),
    (
        "1536",
        "frozen",
        ("--max-num-batched-tokens", "1536", "--long-prefill-token-threshold", "768"),
    ),
    (
        "pr",
        "make-room",
        (
            "--additional-config.make_room.reserve_tokens",
            "0",
            "--additional-config.make_room.budget_cap_tokens",
            "48",
        ),
    ),
)
"""The core's compile configurations: the frozen box's 512-token step cap, vLLM's default 2,048/0, the time-equivalent
cap (docs/HARDWARE.md's R4, its central value; T = B/2), and make-room at the PRs' values (block 3, every E2 round's
make-room point, the batch arm's "on"), which compiles its own graphs at the box's cap: vLLM 0.32.0 hashes
additional_config into the compile key (config/vllm.py), so make-room shares no graph with the frozen box."""
ENGINE_LINES = re.compile(
    r"Selected \S+ for Fp8LinearMethod|native support for FP8|Using \S+ attention backend|"
    r"Using FlashAttention version|Using \S+ KV cache layout|cudaHostRegister|top-p & top-k sampling|"
    r"Building JIT module|torch\.compile takes|init engine .* took|Free memory on device|CUDA error"
)
"""The engine log's lines a warm start keeps: the linear kernel, attention backend, FlashAttention version and KV
layout as selected, pinned memory, the sampler, the JIT and compile work and the start's own account."""
STATUSES = ("ok", "degraded", "refused", "pending")
SOCKET_PATH = 107
"""The longest path a Unix socket may have, in bytes: Linux's sun_path holds 108 with the NUL."""
ZMQ_SOCKET = "flatshare/engine.zmq/" + "0" * 36
"""The longest of the GPU units' socket paths under %t: vLLM's ZMQ sockets, which its unit keeps in a directory of
its own (VLLM_RPC_BASE_PATH, %t/flatshare/engine.zmq, and stt.zmq for the STT), each named by a uuid4."""


class HostError(Exception):
    pass


def _home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def _write(path: Path, text: str, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    with open(
        os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644), "w"
    ) as f:
        f.write(text)
    tmp.replace(path)


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        link.unlink()
    link.symlink_to(target)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


@dataclass
class Node:
    dir: Path
    node: dict
    host: dict

    @classmethod
    def load(cls, directory: Path) -> Node:
        try:
            node = json.loads((directory / "node.json").read_text())
            host = json.loads((directory / "host.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise HostError(
                f"{directory} is not a rendered node with its host's facts: {e}"
            ) from None
        return cls(directory, node, host)

    @property
    def plan(self) -> dict:
        if not self.node.get("plan"):
            raise HostError(
                f"{self.dir}: node.json has no core plan (a profile of fixed values, the box's?)"
            )
        return self.node["plan"]

    def value(self, key: str, default: str = "") -> str:
        return self.node["values"].get(key, default)


def engine_address(host: dict) -> str:
    """The engine's own socket, as the GPU units serve it (%t/flatshare/engine.sock), as an address."""
    return f"unix:{runtime_dir(host) / 'flatshare' / 'engine.sock'}"


def chosen() -> str:
    """FLATSHARE_SUPERVISOR: auto (or unset), flatshare-svc, or the systemd that env.sh records once a stage chose
    it."""
    value = os.environ.get("FLATSHARE_SUPERVISOR") or "auto"
    if value not in SUPERVISORS:
        raise HostError(f"FLATSHARE_SUPERVISOR is auto or flatshare-svc, not {value!r}")
    return value


def overridden() -> bool:
    """Whether flatshare-svc is chosen whatever the facts say (FLATSHARE_SUPERVISOR=flatshare-svc: the operator's
    choice, or an earlier stage's that env.sh keeps); auto or unset lets the facts decide."""
    return chosen() == "flatshare-svc"


def manager_usable(host: dict) -> bool:
    """Whether the host has a user manager that outlives the session: systemd --user, with lingering."""
    return bool(host.get("systemd_user") and host.get("linger"))


def supervisor(host: dict) -> str:
    """What runs the units: deploy/bin/flatshare-svc with FLATSHARE_SUPERVISOR=flatshare-svc; the user manager with
    FLATSHARE_SUPERVISOR=systemd (env.sh's record of an earlier stage's choice, kept even should the facts change: the
    preflight then refuses); with auto, the user manager where the host has one that outlives the session, else
    flatshare-svc."""
    value = chosen()
    if value == "auto":
        return "systemd" if manager_usable(host) else "flatshare-svc"
    return value


def flatshare_svc_runtime() -> Path:
    """deploy/bin/flatshare-svc's runtime directory where the session has no XDG_RUNTIME_DIR."""
    state = os.environ.get("XDG_STATE_HOME") or str(_home() / ".local" / "state")
    return Path(state) / "flatshare-svc" / "run"


def runtime_dir(host: dict) -> Path:
    """%t as the units will see it: XDG_RUNTIME_DIR; else systemd's /run/user/UID, or deploy/bin/flatshare-svc's own."""
    if value := os.environ.get("XDG_RUNTIME_DIR"):
        return Path(value)
    if supervisor(host) == "systemd":
        return Path(f"/run/user/{os.getuid()}")
    return flatshare_svc_runtime()


def write_env(node: Node, home: Path | None = None) -> Path:
    """NODE/env.sh, which every command on this host sources: under flatshare-svc its shims and runtime directory."""
    home = home or _home()
    path = node.dir / "env.sh"
    if supervisor(node.host) == "systemd":
        # The choice is kept: the later stages run the units where this one did (bootstrap.sh).
        _write(
            path,
            "# Sourced before any command on this host (deploy/a100/host.py env writes it): systemd runs the units.\n"
            "export FLATSHARE_SUPERVISOR=systemd\n" + _gpu_line(node.host),
        )
        return path
    state = os.environ.get("XDG_STATE_HOME") or str(home / ".local" / "state")
    run = Path(state) / "flatshare-svc" / "run"
    run.mkdir(mode=0o700, parents=True, exist_ok=True)
    run.chmod(0o700)
    why = (
        "FLATSHARE_SUPERVISOR=flatshare-svc (the operator's, or an earlier stage's choice, kept) runs the units under "
        "deploy/bin/flatshare-svc over the user manager"
        if overridden() and manager_usable(node.host)
        else "no systemd user manager that outlives the session here, so deploy/bin/flatshare-svc runs the units"
    )
    lines = [
        "# Sourced before any command on this host (deploy/a100/host.py env writes it): "
        + why
        + ",",
        "# through the systemctl and journalctl shims.",
        f"export XDG_RUNTIME_DIR=${{XDG_RUNTIME_DIR:-{run}}}",
        f"export PATH={DEPLOY / 'bin' / 'shims'}:$PATH",
        # The choice is kept: the later stages run the units where this one did (bootstrap.sh).
        "export FLATSHARE_SUPERVISOR=flatshare-svc",
    ]
    _write(path, "\n".join(lines) + "\n" + _gpu_line(node.host))
    return path


def _gpu_line(host: dict) -> str:
    """env.sh's lines naming the lent GPU, for the later stages (detect reads it) and the GPU sampler, and the hub
    cache the units read, for the later stages and every script that loads a model's files."""
    # Its UUID once the facts chose it; before the card is visible, the operator's FLATSHARE_GPU_ID as given, which the
    # gpu stage's facts resolve.
    chosen = lent_gpu(host) or (None if host.get("gpu") else os.environ.get("FLATSHARE_GPU_ID"))
    lines = f"export FLATSHARE_GPU_ID={chosen}\n" if chosen else ""
    if (
        os.environ.get("FLATSHARE_ALLOW_MIG") == "1"
    ):  # the operator's choice to measure on a MIG slice
        lines += "export FLATSHARE_ALLOW_MIG=1\n"
    if cache := host.get("hf_hub_cache"):
        lines += f"export HF_HUB_CACHE={shlex.quote(cache)}\n"
    return lines


def active_elsewhere(host: dict) -> tuple[str, list[str]]:
    """The supervisor that does not run this host's units, and those of its units that run under it: an earlier
    stage's, which a split control plane leaves (the gateway's port held by one, the journal read from the other)."""
    if supervisor(host) == "systemd":
        other = "flatshare-svc"
        command = [sys.executable, str(DEPLOY / "bin" / "flatshare-svc"), "systemctl"]
    else:
        other = "systemd"
        shims = str(DEPLOY / "bin" / "shims")
        path = os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep) if p != shims
        )
        manager = shutil.which("systemctl", path=path) if host.get("systemd_user") else None
        if manager is None:
            return other, []
        command = [manager]
    done = _run([*command, "--user", "is-active", *UNITS])
    states = done.stdout.split()
    return other, [unit for unit, state in zip(UNITS, states, strict=False) if state == "active"]


# units


def drop_ins(node: Node) -> dict[str, str]:
    """This host's drop-in for each unit that needs one: the control plane's cores and, where the kernel refuses
    unprivileged user namespaces, the voice plane's and the GPU servers' mark for the host's network namespace."""
    plan = node.plan
    header = "# Single host (deploy/a100/host.py units, from the node's core plan in node.json).\n"
    cores = lambda cpus: f"CPUAffinity=\nCPUAffinity={cpus}\n"  # noqa: E731
    out = {
        "flatshare-gateway.service": header + "[Service]\n" + cores(plan["gateway"]),
        "flatshare-livekit.service": header + "[Service]\n" + cores(plan["voice"]),
        "flatshare-voice-worker.service": header
        + "# No tunnel: the STT and the TTS are this host's, on their own sockets (flatshare-stt and flatshare-tts).\n"
        "[Service]\nEnvironment=FLATSHARE_STT_UDS=%t/flatshare/stt.sock\n"
        "Environment=FLATSHARE_TTS_UDS=%t/flatshare/tts.sock\n" + cores(plan["voice"]),
    }
    if node.host.get("userns") is not True:
        out["flatshare-voicenet.service"] = (
            header
            + "# The kernel refuses unprivileged user namespaces here: the voice plane runs in the host's "
            "network\n# namespace (deploy/bin/voicenet-holder), which the preflight reports.\n"
            "[Service]\nEnvironment=FLATSHARE_VOICENET=host\n"
        )
        for server in PRIVATE_SERVERS:
            out[server] = (
                header
                + "# The kernel refuses unprivileged user namespaces here: the GPU server runs in the host's "
                "network\n# namespace (deploy/bin/in-private-net), which the preflight reports.\n"
                "[Service]\nEnvironment=FLATSHARE_PRIVATE_NET=host\n"
            )
    return out


def install_units(node: Node, units: Path | None = None, systemctl: str = "systemctl") -> list[str]:
    """Links the units and the drop-ins; returns the links made."""
    units = units or _home() / ".config" / "systemd" / "user"
    links: list[tuple[Path, Path]] = []
    for path in sorted((node.dir / "systemd").iterdir()):
        if path.is_file():
            links.append((path, units / path.name))
    for name in CONTROL:
        links.append((CYAN / "systemd" / name, units / name))
    wanted = drop_ins(node)
    for unit, text in wanted.items():
        conf = node.dir / "single-host" / f"{unit}.d" / "single-host.conf"
        _write(conf, text)
        links.append((conf, units / f"{unit}.d" / "single-host.conf"))
    for variant in WORKER_VARIANTS:
        conf = CYAN / "variants" / variant / "flatshare-voice-worker.conf"
        links.append((conf, units / "flatshare-voice-worker.service.d" / f"variant-{variant}.conf"))
    for target, link in links:
        _link(target, link)
    # Links to units the repository no longer has (the tunnels' before flatshare-tunnel@) go, as
    # deploy/cyan/install.sh's.
    for link in units.glob("*"):
        if link.is_symlink() and not link.exists():
            target = Path(os.readlink(link))
            if target.parent in (CYAN / "systemd", node.dir / "systemd"):
                link.unlink()
    # A drop-in this host no longer needs (the voice plane's, once namespaces are allowed) goes.
    for stale in units.glob("*.d/single-host.conf"):
        if stale.parent.name.removesuffix(".d") not in wanted:
            stale.unlink()
    done = _run([systemctl, "--user", "daemon-reload"])
    if done.returncode:
        raise HostError(f"systemctl --user daemon-reload: {done.stderr.strip()}")
    return [str(link) for _, link in links]


# config


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise HostError(f"deploy/cyan/gateway.example.toml changed: no single {what} ({old!r})")
    return text.replace(old, new)


def gateway_toml(home: Path, runtime: Path, keys: dict[str, str]) -> str:
    """cyan's gateway.toml for this host: its home's files, its runtime directory's sockets, the engine's own."""
    text = (CYAN / "gateway.example.toml").read_text()
    head, sep, rest = text.partition("\n\n")
    if not head.startswith("# flatshare-gateway on cyan"):
        raise HostError("deploy/cyan/gateway.example.toml changed: its header is not cyan's")
    text = (
        "# flatshare-gateway on this single host, written by deploy/a100/host.py config from deploy/cyan/gateway.example.toml\n"
        "# (mode 600). It holds the tenants' key digests; the keys are in gateway-keys.env. Every key is described in\n"
        "# crates/flatshare-gateway/README.md." + sep + rest
    )
    text = _replace_once(
        text,
        "# vLLM on Phastform, through its tunnel's socket (flatshare-tunnel@phastform; cyan's deploy user's "
        'uid, 1007, as below).\nupstream = "unix:/run/user/1007/flatshare/phastform-engine.sock"\n',
        "# vLLM on this host, on its own socket (flatshare-engine.service's %t/flatshare/engine.sock): no tunnel.\n"
        f"upstream = {json.dumps('unix:' + str(runtime / 'flatshare' / 'engine.sock'))}\n",
        "upstream",
    )
    text = _replace_once(
        text,
        '"<home>/.config/flatshare/cache-salt-secret"',
        json.dumps(str(home / ".config/flatshare/cache-salt-secret")),
        "cache_salt_secret_file",
    )
    text = _replace_once(
        text,
        '"<home>/.config/flatshare/metrics-key"',
        json.dumps(str(home / ".config/flatshare/metrics-key")),
        "metrics_key_file",
    )
    text = _replace_once(
        text,
        'unix = "/run/user/1007/flatshare/gateway.sock"',
        f"unix = {json.dumps(str(runtime / 'flatshare' / 'gateway.sock'))}",
        "voice socket",
    )
    # The rest of cyan's runtime directory, in the commented examples, is this host's too.
    text = text.replace("/run/user/1007/", f"{runtime}/")
    for tenant in ("agents", "bench"):
        digest = hashlib.sha256(
            keys[f"FLATSHARE_GATEWAY_{tenant.upper()}_KEY"].encode()
        ).hexdigest()
        text = _replace_once(
            text, f'"<sha256 of the {tenant} key>"', f'"{digest}"', f"{tenant} key"
        )
    return text


def _read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.startswith("#"):
            env[key.strip()] = value.strip().strip('"')
    return env


def write_config(node: Node, home: Path | None = None) -> list[str]:
    """Writes the control plane's configuration where it is missing, and e2-host.toml; returns the files written."""
    home = home or _home()
    config = home / ".config" / "flatshare"
    written = []
    salt = config / "cache-salt-secret"
    if not salt.exists():
        _write(salt, secrets.token_hex(32) + "\n", private=True)
        written.append(salt)
    metrics = config / "metrics-key"  # the key the gateway's TCP listener asks for on GET /metrics
    if not metrics.exists():
        _write(metrics, secrets.token_hex(32) + "\n", private=True)
        written.append(metrics)
    keys_file = config / "gateway-keys.env"
    if not keys_file.exists():
        keys = {
            f"FLATSHARE_GATEWAY_{t}_KEY": secrets.token_urlsafe(32) for t in ("AGENTS", "BENCH")
        }
        _write(
            keys_file,
            "# The gateway tenants' keys on this host (deploy/a100/host.py config); gateway.toml holds their digests.\n"
            + "".join(f"{k}={v}\n" for k, v in keys.items()),
            private=True,
        )
        written.append(keys_file)
    gateway = config / "gateway.toml"
    if not gateway.exists():
        _write(
            gateway, gateway_toml(home, runtime_dir(node.host), _read_env(keys_file)), private=True
        )
        written.append(gateway)
    livekit = config / "livekit.env"
    if not livekit.exists():
        key = "flatshare" + secrets.token_hex(6)
        value = secrets.token_urlsafe(32).replace("-", "").replace("_", "")
        _write(
            livekit,
            f'LIVEKIT_KEYS="{key}: {value}"\nLIVEKIT_API_KEY={key}\nLIVEKIT_API_SECRET={value}\n'
            "LIVEKIT_URL=ws://127.0.0.1:17880\n",
            private=True,
        )
        written.append(livekit)
    plan = node.plan
    host_file = config / "e2-host.toml"
    _write(
        host_file,
        "# flatshare-bench e2 run on this single host (deploy/a100/host.py config, from the node's core plan): "
        "the engine on\n"
        "# its own socket, no tunnel; the gateway's socket; the load generators' cores; the node's profile and its\n"
        "# preflight; and the node's name, as campaign.sh --local calls it.\n"
        f'gateway_url = "{GATEWAY_URL}"\n'
        f"gateway_socket = {json.dumps('unix:' + str(runtime_dir(node.host) / 'flatshare' / 'gateway.sock'))}\n"
        f"engine_url = {json.dumps(engine_address(node.host))}\n"
        'node = "localhost"\n'
        f'aiperf_cores = "{plan["aiperf"]}"\ncaller_cores = "{plan["callers"]}"\n'
        f'profile = "{node.node["profile"]}"\npreflight = {json.dumps(str(node.dir / "preflight.json"))}\n',
    )
    written.append(host_file)
    return [str(path) for path in written]


# jit-check


def unit_environment(unit: Path, home: Path) -> dict[str, str]:
    """A rendered unit's Environment= settings, %h expanded."""
    env = {}
    for line in unit.read_text().splitlines():
        if line.startswith("Environment="):
            for assignment in shlex.split(
                line.removeprefix("Environment=").replace("%h", str(home))
            ):
                key, sep, value = assignment.partition("=")
                if sep:
                    env[key] = value
    return env


def arch(node: Node, override: str | None = None) -> str:
    """The GPU's compute capability as FlashInfer's FLASHINFER_CUDA_ARCH_LIST takes it: the card's, else the
    profile's (before the GPU is visible)."""
    cap = (
        override
        or (node.host.get("gpu") or {}).get("compute_cap")
        or node.value("profile.compute_cap")
    )
    if not re.fullmatch(r"\d+\.\d+", cap or ""):
        raise HostError(
            "no compute capability: the host shows no GPU and the profile names none (--arch)"
        )
    return cap


BUILD = """
import json, sys, time
from flashinfer.jit import env
from flashinfer.jit.sampling import gen_sampling_module
spec = gen_sampling_module()
start = time.monotonic()
spec.build(verbose=False)
print(json.dumps({"library": str(spec.get_library_path()), "workspace": str(env.FLASHINFER_WORKSPACE_DIR),
                  "build_s": round(time.monotonic() - start, 1)}))
"""


def jit_check(node: Node, home: Path | None = None, cap: str | None = None) -> dict:
    """Builds FlashInfer's sampler with the engine unit's environment; writes cache/jit-check.json."""
    home = home or _home()
    unit = node.dir / "systemd" / "flatshare-engine.service"
    env = {"HOME": str(home), "LANG": "C.UTF-8"} | unit_environment(unit, home)
    env["FLASHINFER_CUDA_ARCH_LIST"] = arch(node, cap)
    python = home / "flatshare" / "venvs" / "engine" / "bin" / "python"
    started = time.time()
    try:
        done = _run([str(python), "-c", BUILD], env=env, timeout=3600)
        ok, output, error = done.returncode == 0, done.stdout, done.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        ok, output, error = False, "", str(e)
    result = {
        "time": started,
        "arch": env["FLASHINFER_CUDA_ARCH_LIST"],
        "cuda_home": env.get("CUDA_HOME"),
        "seconds": round(time.time() - started, 1),
        "ok": ok,
    }
    if ok:
        result |= json.loads(output.strip().splitlines()[-1])
    else:
        result["error"] = "\n".join(error.strip().splitlines()[-12:])
    _write(home / "flatshare" / "cache" / "jit-check.json", json.dumps(result, indent=2) + "\n")
    return result


def jit_current(node: Node, home: Path | None = None) -> tuple[bool, str]:
    """Whether cache/jit-check.json's build is for the compiler the engine unit names (its CUDA_HOME) and the card's
    compute capability, and if not, why: the gpu stage can change both (a cu129 lock and CUDA 12.9's toolkit where
    the stage rendered cu130; the card's own capability where the stage had the profile's)."""
    home = home or _home()
    record = _json(home / "flatshare" / "cache" / "jit-check.json")
    if record is None:
        return False, "not built"
    wanted = unit_environment(node.dir / "systemd" / "flatshare-engine.service", home).get(
        "CUDA_HOME", ""
    )
    try:
        cap = arch(node)
    except HostError as e:
        return False, str(e)
    stale = []
    if (record.get("cuda_home") or "").replace("%h", str(home)) != wanted:
        stale.append(f"built with {record.get('cuda_home')}, where the engine unit names {wanted}")
    if record.get("arch") != cap:
        stale.append(f"built for {record.get('arch')}, where the card is {cap}")
    return not stale, "; ".join(stale)


# warm


class _UnixConnection(http.client.HTTPConnection):
    """HTTP on a server's Unix socket."""

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def _get(address: str, path: str, timeout: float = 10) -> str | None:
    """GET ``path`` from the server at ``address`` (``http://host:port``, or ``unix:`` and its socket's path, as
    flatshare-bench's addresses): the body of a 2xx answer, else None."""
    try:
        if address.startswith("unix:"):
            connection: http.client.HTTPConnection = _UnixConnection(
                address.removeprefix("unix:"), timeout
            )
        else:
            url = urllib.parse.urlsplit(address)
            connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read().decode()
        finally:
            connection.close()
    except (OSError, http.client.HTTPException, ValueError):
        return None
    return body if 200 <= response.status < 300 else None


def engine_facts(address: str) -> dict:
    """What the running engine at ``address`` says of its scheduler and cache: async scheduling, the linear
    backend (/server_info, the measurement drop-in's), the block size and the pool's blocks."""
    facts: dict = {}
    if info := _get(address, "/server_info?config_format=json"):
        try:
            config = json.loads(info).get("vllm_config", {})
            facts["async_scheduling"] = (config.get("scheduler_config") or {}).get(
                "async_scheduling"
            )
            facts["linear_backend"] = (config.get("kernel_config") or {}).get("linear_backend")
        except (json.JSONDecodeError, AttributeError):
            pass
    metrics = _get(address, "/metrics") or ""
    if m := re.search(r"vllm:cache_config_info\{([^}]*)\}", metrics):
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        facts["block_size"] = (
            int(labels["block_size"]) if labels.get("block_size", "").isdigit() else None
        )
        facts["num_gpu_blocks"] = (
            int(labels["num_gpu_blocks"]) if labels.get("num_gpu_blocks", "").isdigit() else None
        )
    return facts


def configurations(path: Path | None) -> list[tuple[str, str, tuple[str, ...]]]:
    """WARM, or one configuration a line from `path`: NAME VARIANT [FLAG...]; # starts a comment."""
    if path is None:
        return list(WARM)
    found = []
    for line in path.read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if words:
            if len(words) < 2:
                raise HostError(f"{path}: {line!r} is not NAME VARIANT [FLAG...]")
            found.append((words[0], words[1], tuple(words[2:])))
    return found


def warm(
    node: Node,
    configs: list[tuple[str, str, tuple[str, ...]]],
    home: Path | None = None,
    engine: str | None = None,
) -> dict:
    """Starts the GPU units once for each configuration (variant.sh use), in order, with the measurement drop-in
    on (variant.sh measure on); writes cache/warm.json."""
    home = home or _home()
    engine = engine or engine_address(node.host)
    measure = _run(["bash", str(node.dir / "variant.sh"), "measure", "on"], timeout=2400)
    if measure.returncode:
        said = (measure.stderr or measure.stdout).strip().splitlines()
        raise HostError(f"variant.sh measure on: {said[-1] if said else 'failed'}")
    path = home / "flatshare" / "cache" / "warm.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        record = {"configurations": {}}
    for name, variant, flags in configs:
        started = time.time()
        done = _run(["bash", str(node.dir / "variant.sh"), "use", variant, *flags], timeout=2400)
        ready = time.time()
        journal = _run(
            [
                "journalctl",
                "--user",
                "--unit",
                "flatshare-engine",
                "--since",
                f"@{int(started)}",
                "-o",
                "cat",
            ]
        )
        lines = [line for line in journal.stdout.splitlines() if ENGINE_LINES.search(line)]
        entry = {
            "variant": variant,
            "flags": list(flags),
            "started": started,
            "ready_s": round(ready - started, 1),
            "ok": done.returncode == 0,
            "log": lines[-40:],
        }
        if done.returncode == 0:
            entry |= engine_facts(engine)
        else:
            entry["error"] = "\n".join((done.stderr or done.stdout).strip().splitlines()[-8:])
        record["configurations"][name] = entry
        record["time"] = time.time()
        _write(path, json.dumps(record, indent=2) + "\n")
        print(
            f"warm: {name} ({variant} {' '.join(flags)}) {'ready' if entry['ok'] else 'FAILED'} after {entry['ready_s']} s"
        )
    return record


# preflight


@dataclass
class Check:
    check: str
    status: str
    value: str
    reason: str = ""
    effect: str = ""


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _mount(path: Path, mounts: str) -> tuple[str, str]:
    """The mount point and file system type that hold `path`, from /proc/mounts' text."""
    best = ("/", "unknown")
    resolved = str(path.resolve())
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            point = parts[1].replace("\\040", " ")
            if (resolved == point or resolved.startswith(point.rstrip("/") + "/")) and len(
                point
            ) >= len(best[0]):
                best = (point, parts[2])
    return best


def _free(path: Path) -> int | None:
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        s = os.statvfs(path)
    except OSError:
        return None
    return s.f_bavail * s.f_frsize


def _public_addresses() -> list[str]:
    """This host's addresses that are neither loopback, private, link-local nor shared (CGNAT)."""
    found = set()
    try:
        for *_, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
            found.add(sockaddr[0])
    except OSError:
        pass
    done = shutil.which("ip") and _run(["ip", "-o", "addr", "show"])
    if done and done.returncode == 0:
        found |= set(re.findall(r"inet6? ([0-9a-f.:]+)/", done.stdout))
    public = []
    for text in sorted(found):
        try:
            address = ipaddress.ip_address(text)
        except ValueError:
            continue
        if address.is_global:
            public.append(text)
    return public


def hub_cache(host: dict, home: Path) -> Path:
    """The Hugging Face hub cache the units read: the one the facts recorded (bootstrap.sh resolves it once, from the
    operator's HF_HOME or HF_HUB_CACHE at the first stage); the default under HOME for facts from before."""
    return Path(host.get("hf_hub_cache") or home / ".cache" / "huggingface" / "hub")


def _hf_snapshot(hub: Path, model: str, revision: str) -> bool:
    """Whether the hub cache holds `model` at `revision`, as the units' offline load reads it."""
    snapshot = hub / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    return snapshot.is_dir() and any(snapshot.iterdir())


DATA = {
    "weka-x32-long": ("*.json", "traces"),
    "weka-x32": ("*.json", "traces"),
    "callers": ("*.wav", "clips"),
}
"""The campaign's data under ~/flatshare/data (deploy/a100/push.sh copies them): each directory's files that the
campaign reads, and what they are. The callers' corpus also needs its manifest.json (flatshare-voice bots reads it)."""


def data_check(root: Path, data: dict[str, tuple[str, str]] | None = None) -> Check:
    """The preflight's corpus row: every directory of ``data`` under ROOT/data holds its files, and each resolves to
    a regular file this user can read. A copy of a workstation's links (long_traces.py links its traces by their
    absolute paths) dangles here, and AIPerf would fail on it only at the first agent point, hours in."""
    found, wrong = [], []
    for name, (pattern, kind) in (DATA if data is None else data).items():
        directory = root / "data" / name
        if not directory.is_dir():
            wrong.append(f"{name} missing")
            continue
        files = sorted(directory.glob(pattern))
        if name == "callers":
            files.append(directory / "manifest.json")
        dangling = [f.name for f in files if not (f.is_file() and os.access(f, os.R_OK))]
        if dangling:
            wrong.append(
                f"{name}: {len(dangling)} of {len(files)} do not resolve to a readable file ({dangling[0]}"
                + (", ..." if len(dangling) > 1 else "")
                + ")"
            )
        elif not files:
            wrong.append(f"{name}: no {kind} ({pattern})")
        else:
            found.append(f"{name} {len(files) - (name == 'callers')} {kind}")
    if wrong:
        return Check(
            "Corpus and callers",
            "refused",
            "; ".join(wrong),
            "under ~/flatshare/data: deploy/a100/push.sh copies them, as files (links into the workstation's paths "
            "dangle here)",
        )
    return Check("Corpus and callers", "ok", ", ".join(found))


def _yaml_model(path: Path) -> tuple[str, str] | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    model = re.search(r"^model: (\S+)$", text, re.M)
    revision = re.search(r"^revision: (\S+)$", text, re.M)
    return (model.group(1), revision.group(1)) if model and revision else None


def lent_gpu(host: dict) -> str | None:
    """The UUID of the GPU this host lends, once its facts chose one (profile.py detect, FLATSHARE_GPU_ID)."""
    card = host.get("gpu") or {}
    return card.get("uuid") if card.get("chosen") else None


def _gpu_processes(uuid: str | None = None) -> list[str] | None:
    done = shutil.which("nvidia-smi") and _run(
        [
            "nvidia-smi",
            *(["-i", uuid] if uuid else []),
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ]
    )
    if not done or done.returncode:
        return None
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


HEADERS_PROBE = (
    "import os, sysconfig; include = sysconfig.get_paths()['include']; "
    "print(include); print(os.path.exists(os.path.join(include, 'Python.h')))"
)


def python_headers(python: Path) -> tuple[bool, str] | None:
    """Whether ``python`` has its headers (Python.h in its sysconfig include directory), which Triton compiles its
    CUDA launcher against at the engine's first start, and that directory; None where there is no such interpreter.
    uv's own Python has them; a system Python 3.12 has them only with its -dev package."""
    if not python.exists():
        return None
    try:
        done = _run([str(python), "-c", HEADERS_PROBE], timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    found = done.stdout.split("\n")
    if done.returncode or len(found) < 2:
        return False, (done.stderr.strip().splitlines() or ["no output"])[-1]
    return found[1].strip() == "True", found[0].strip()


def _series(words: list[str], conjunction: str = "and") -> str:
    """``words`` as a sentence lists them: "a", "a and b", "a, b and c"."""
    return f"{', '.join(words[:-1])} {conjunction} {words[-1]}" if len(words) > 1 else words[0]


def outside_linker(node: Node, home: Path) -> tuple[str, list[str]] | None:
    """The directory of the driver's libcuda.so.1 where the facts found that the dynamic linker does not search it
    (profile.py detect's driver_libs: Colab's container keeps it in /usr/lib64-nvidia, which only its shells'
    LD_LIBRARY_PATH reaches), and the GPU units whose environment does not name it on LD_LIBRARY_PATH, as theirs must,
    having none of the operator's; None where the linker finds it, or the facts found none."""
    libs = node.host.get("driver_libs") or {}
    if not libs.get("path") or libs.get("linker"):
        return None
    unnamed = []
    for unit in PRIVATE_SERVERS:
        path = unit_environment(node.dir / "systemd" / unit, home).get("LD_LIBRARY_PATH", "")
        if libs["path"] not in path.split(":"):
            unnamed.append(unit.removesuffix(".service"))
    return libs["path"], unnamed


TORCH_PROBE = (
    "import torch; torch.cuda.init(); "
    "print(torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
)
NO_DRIVER = "Found no NVIDIA driver"
"""torch's error where the CUDA runtime cannot load the driver's libcuda.so.1 (cudaDriverGetVersion gives 0)."""


def torch_on_gpu(node: Node, home: Path) -> tuple[bool, str]:
    """Whether torch in the engine venv initializes CUDA with the engine unit's environment, and what it said."""
    env = {"HOME": str(home)} | unit_environment(
        node.dir / "systemd" / "flatshare-engine.service", home
    )
    python = home / "flatshare" / "venvs" / "engine" / "bin" / "python"
    try:
        done = _run([str(python), "-c", TORCH_PROBE], env=env, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if done.returncode == 0:
        return True, done.stdout.strip()
    return False, (done.stderr.strip().splitlines() or ["no output"])[-1]


def preflight(
    node: Node,
    gpu: bool = False,
    home: Path | None = None,
    data: dict[str, tuple[str, str]] | None = None,
    gateway_url: str = GATEWAY_URL,
) -> list[Check]:
    """Every check, in docs/HARDWARE.md's order (the preflight)."""
    home = home or _home()
    root = home / "flatshare"
    host, values = node.host, node.node["values"]
    card = host.get("gpu") or {}
    warmed = (_json(root / "cache" / "warm.json") or {}).get("configurations", {})
    started = [entry for entry in warmed.values() if entry.get("ok")]
    log = "\n".join(line for entry in started for line in entry.get("log", []))
    checks: list[Check] = []
    add = lambda *args: checks.append(Check(*args))  # noqa: E731
    no_start = "the engine's first start (bootstrap.sh warm) shows it"

    # The card and its precision path.
    cap = card.get("compute_cap")
    if not cap:
        add(
            "CC and precision path",
            "pending",
            "no GPU visible",
            "the GPU checks run once nvidia-smi shows the card",
        )
    else:
        major, minor = (int(x) for x in cap.split("."))
        kernel = re.search(r"Selected (\S+) for Fp8LinearMethod", log)
        seen = f"; the engine selected {kernel.group(1)}" if kernel else ""
        if (major, minor) < (7, 5):
            add(
                "CC and precision path",
                "refused",
                f"sm{major}{minor}",
                "vLLM 0.32.0 needs compute capability 7.5 or newer",
            )
        else:
            # The path is the kernel the engine's start selected; before a start, the one the card is expected to
            # take: vLLM 0.32.0's block-FP8 kernels ahead of Marlin need sm90, so below it this checkpoint goes to
            # Marlin's W8A16 (kernels/linear/__init__.py), from it to FP8 W8A8.
            selected = kernel.group(1) if kernel else None
            expected = "" if selected else ", expected (the engine's first start shows its kernel)"
            effect = "within-card comparisons unchanged; across cards only in regime coordinates"
            if (not selected and major < 9) or (selected and "marlin" in selected.lower()):
                add(
                    "CC and precision path",
                    "degraded",
                    f"sm{major}{minor}{seen}",
                    "W8A16 through Marlin at the BF16 rate"
                    + (" (FP16 activations)" if major < 8 else "")
                    + expected,
                    "the per-token cost is the BF16 path's; " + effect,
                )
            elif major >= 9:
                add("CC and precision path", "ok", f"sm{major}{minor}: FP8 W8A8{seen}{expected}")
            else:
                add(
                    "CC and precision path",
                    "degraded",
                    f"sm{major}{minor}{seen}",
                    f"FP8 through {selected}: neither Marlin's W8A16, which the forecast models below sm90, nor the "
                    "box's W8A8",
                    "the per-token cost is that kernel's; " + effect,
                )
        if major >= 8:
            add("BF16", "ok", "CC >= 8.0")
        else:
            add(
                "BF16",
                "degraded",
                f"sm{major}{minor}",
                "FP16 activations",
                "numerics differ from the box; unverified for quality",
            )
    backend = re.search(r"Using (\S+) attention backend", log)
    layout = re.search(r"Using (\S+) KV cache layout", log)
    fa = re.search(r"Using FlashAttention version (\d)", log)
    if not backend:
        add("Attention backend and KV layout", "pending", "not started", no_start)
    else:
        seen = f"{backend.group(1)}{' ' + fa.group(1) if fa else ''}, {layout.group(1) if layout else 'layout not logged'}"
        good = (
            backend.group(1) in ("FLASH_ATTN", "FLASHINFER")
            and layout
            and layout.group(1) == "LBHNC"
        )
        add(
            "Attention backend and KV layout",
            "ok" if good else "degraded",
            seen,
            "" if good else f"{seen}, not the box's FA on LBHNC",
            "" if good else "attention's cost; recorded",
        )

    # The driver, the install path and the compilers.
    install = node.node.get("install") or {}
    built = (
        (root / "venvs" / "engine" / "flatshare-install-path").read_text().strip()
        if (root / "venvs" / "engine" / "flatshare-install-path").exists()
        else None
    )
    path = install.get("path")
    value = f"driver {card.get('driver') or 'not seen'}, {path or 'no install path'}" + (
        f" (venv: {built})" if built else ""
    )
    # The driver's libraries where the linker does not search them, which every GPU unit must name: their environment
    # has no LD_LIBRARY_PATH of the operator's.
    outside = outside_linker(node, home)
    unsearched = (
        outside
        and f"the driver's libcuda.so.1 is in {outside[0]}, which the linker does not search"
    )
    on_path = (
        f"; the driver's libraries in {outside[0]}, which the linker does not search, on the GPU units' "
        "LD_LIBRARY_PATH"
        if outside and not outside[1]
        else ""
    )
    if built and path and built != path:
        add(
            "Driver and install path",
            "refused",
            value + on_path,
            f"the engine venv holds the {built} build but the driver needs {path}: bootstrap.sh gpu",
        )
    elif outside and outside[1]:
        add(
            "Driver and install path",
            "refused",
            value,
            f"{unsearched}, and {_series(outside[1])} {'do' if len(outside[1]) > 1 else 'does'} not name it on "
            "LD_LIBRARY_PATH: bootstrap.sh stage or gpu renders the GPU units with it",
        )
    elif not card.get("driver"):
        reason = install.get("reason", "no driver seen")
        add("Driver and install path", "pending", value + on_path, reason)
    elif path == "cu130":
        add("Driver and install path", "ok", value + ": the box's lock" + on_path)
    else:
        add(
            "Driver and install path",
            "degraded",
            value + on_path,
            install.get("reason", ""),
            "the same vLLM and torch versions on another CUDA runtime; recorded",
        )
    if gpu:
        ok, said = torch_on_gpu(node, home)
        if ok:
            why = ""
        elif outside and "flatshare-engine" in outside[1]:
            why = (
                f"{unsearched}, and the engine unit's environment does not name it on LD_LIBRARY_PATH: "
                "bootstrap.sh gpu renders the GPU units with it"
            )
        elif NO_DRIVER in said and "driver_libs" in host and not host["driver_libs"]:
            where = _series(["the linker's cache", "LD_LIBRARY_PATH", *profile.DRIVER_DIRS], "or")
            why = (
                f"no libcuda.so.1 where the facts look ({where}): the driver's libraries are elsewhere, or not on "
                "this host (a container started without the NVIDIA runtime); with their directory on "
                "LD_LIBRARY_PATH, bootstrap.sh gpu renders the GPU units with it"
            )
        else:
            why = (
                "torch.cuda.init() fails in the engine venv with the engine unit's environment (errors 803 and 804: "
                "the CUDA runtime and the driver do not match)"
            )
        add("torch on the GPU", "ok" if ok else "refused", said, why)
    jit = _json(root / "cache" / "jit-check.json")
    if jit is None:
        add(
            "FlashInfer's JIT",
            "pending",
            "not built",
            "bootstrap.sh stage builds the sampler (host.py jit-check)",
        )
    elif jit.get("ok") and not (current := jit_current(node, home))[0]:
        add(
            "FlashInfer's JIT",
            "refused" if gpu else "pending",
            f"the sampler for sm{jit['arch'].replace('.', '')} with {jit.get('cuda_home')}, stale: {current[1]}",
            "the build checked another compiler or card than the engine's first start will use: bootstrap.sh gpu "
            "builds it again (host.py jit-check --if-stale)",
        )
    elif jit.get("ok"):
        add(
            "FlashInfer's JIT",
            "ok",
            f"the sampler for sm{jit['arch'].replace('.', '')} in {jit['seconds']} s with {jit.get('cuda_home')}",
        )
    else:
        add(
            "FlashInfer's JIT",
            "refused",
            f"the sampler for sm{jit['arch'].replace('.', '')} with {jit.get('cuda_home')}: {(jit.get('error') or '').splitlines()[-1:]}",
            "the engine's first start would fail where it JIT-compiles the sampler; fix the compiler, or set "
            "VLLM_USE_FLASHINFER_SAMPLER=0 in a flatshare-engine drop-in (PyTorch's sampler: degraded, recorded)",
        )
    compilers = [name for name in ("cc", "cxx") if not host.get(name)]
    if compilers:
        add(
            "Host compilers",
            "refused",
            f"missing: {', '.join(compilers)}",
            "Triton builds its launchers with a C compiler and nvcc needs a host C++ compiler: install gcc and g++ (build-essential)",
        )
    else:
        add("Host compilers", "ok", f"{host['cc']}, {host['cxx']}")
    headers = python_headers(root / "venvs" / "engine" / "bin" / "python")
    if headers is None:
        add("Python headers", "pending", "no engine venv", "bootstrap.sh stage makes it")
    elif headers[0]:
        add("Python headers", "ok", f"Python.h in {headers[1]}")
    else:
        add(
            "Python headers",
            "refused",
            f"no Python.h in {headers[1]}",
            "Triton compiles its CUDA launcher against Python.h at the engine's first start, which would fail: "
            "bootstrap.sh stage makes the engine venv anew from uv's own Python 3.12, which has its headers",
        )

    # The caches.
    cache = root / "cache"
    mounts = Path(os.environ.get("FLATSHARE_PROCFS", "/proc")) / "mounts"
    point, fstype = _mount(cache, mounts.read_text() if mounts.exists() else "")
    free = _free(cache)
    wanted = [name for name, *_ in WARM] + sorted(set(warmed) - {name for name, *_ in WARM})
    cold = [name for name in wanted if not (warmed.get(name) or {}).get("ok")]
    value = (
        f"{fstype} at {point}, {free / GIB:.0f} GiB free"
        if free is not None
        else f"{fstype} at {point}"
    )
    if free is not None and free < 2 * GIB:
        add(
            "Compile and JIT caches",
            "refused",
            value,
            "under 2 GiB free for the compile and JIT caches",
        )
    elif fstype in ("tmpfs", "ramfs"):
        add(
            "Compile and JIT caches",
            "degraded",
            value,
            "the caches go with a reboot",
            "start time only",
        )
    elif cold:
        add(
            "Compile and JIT caches",
            "degraded",
            value + f"; cold: {', '.join(cold)}",
            "the first start of each cold compile configuration compiles (the box: 531.8 s against 88.8 s warm)",
            "start time only; budgeted in block 0",
        )
    elif free is not None and free < 10 * GIB:
        add(
            "Compile and JIT caches",
            "degraded",
            value,
            "under 10 GiB free [estimate of what the caches take]",
            "start time only",
        )
    else:
        add("Compile and JIT caches", "ok", value + "; warm: " + ", ".join(wanted))

    # The CPU tier, /dev/shm and memory.
    pool = int(values.get("engine.kv_cache_memory", "0"))
    tier = int(values.get("engine.cpu_bytes_to_use", "0") or 0)
    shm = host.get("shm") or {}
    ratio = tier / pool if pool else 0
    staged = (
        (_json(node.dir / "staged.json") or {}).get("values", {}).get("engine.cpu_bytes_to_use")
    )
    moved = staged is not None and str(staged) != str(tier)
    value = (
        f"{tier / GIB:.0f} GiB, {ratio:.1f} x the pool"
        + (f", where the stage rendered {int(staged) / GIB:.0f} GiB" if moved else "")
        + f"; /dev/shm {shm.get('total_bytes', 0) / GIB:.0f} GiB"
        + (" (a container)" if host.get("container") else "")
    )
    if moved:
        add(
            "CPU tier and /dev/shm",
            "refused",
            value,
            "R3 bounds the tier by /dev/shm's free space and the RAM available, which something holds now (an engine "
            "left running, another tenant): the core runs the staged tier for the whole campaign; free them and run "
            "bootstrap.sh gpu again (or bootstrap.sh stage, to stage this tier)",
        )
    elif ratio >= 4:
        add("CPU tier and /dev/shm", "ok", value)
    else:
        add(
            "CPU tier and /dev/shm",
            "degraded",
            value,
            f"CPU tier at {ratio:.1f} x the pool ({values.get('derived.cpu_tier_note', '')})",
            "fewer restores and more recompute at high pressure; the ratio is in every record",
        )
    add(
        "Shmem huge pages",
        "ok",
        str(host.get("thp_shmem")),
        "",
        "recorded; start time only (the box's tier takes about 14 s of each start)",
    )
    if not started:
        add("Pinned memory", "pending", "not started", no_start)
    elif "cudaHostRegister" in log:
        add(
            "Pinned memory",
            "degraded",
            "cudaHostRegister failed",
            "pageable CPU tier",
            "slower restores; the calibration measures them",
        )
    else:
        add("Pinned memory", "ok", "no cudaHostRegister failure in the engine log")
    numa = node.plan.get("engine_numa") if node.node.get("plan") else None
    nodes = (host.get("memory") or {}).get("nodes") or {}
    local = (nodes.get(str(numa)) or {}).get("free_bytes")
    if local is not None and len(nodes) > 1 and local < tier:
        add(
            "Host RAM",
            "degraded",
            f"{local / GIB:.0f} GiB free on the GPU's node {numa}",
            "remote-NUMA tier",
            "the box measured a remote tier within 1% of pinned bandwidth; recorded",
        )
    else:
        add(
            "Host RAM",
            "ok",
            f"{(host.get('memory') or {}).get('available_bytes', 0) / GIB:.0f} GiB available",
        )

    # The engine as it started.
    facts = next((entry for entry in started if entry.get("async_scheduling") is not None), None)
    if facts is None:
        add("Async scheduling", "pending", "not started", no_start)
    elif facts["async_scheduling"]:
        add("Async scheduling", "ok", "true")
    else:
        add(
            "Async scheduling",
            "degraded",
            "false",
            "synchronous scheduling: make-room loads its Scheduler, not AsyncScheduler",
            "both arms change together; recorded",
        )
    sized = next((entry for entry in started if entry.get("block_size")), None)
    if sized is None:
        add("Block size", "pending", "not started", no_start)
    else:
        add(
            "Block size",
            "ok" if sized["block_size"] == 16 else "degraded",
            f"{sized['block_size']} ({sized.get('num_gpu_blocks')} blocks)",
            "" if sized["block_size"] == 16 else f"block size {sized['block_size']}",
            "" if sized["block_size"] == 16 else "make-room's token counts round to the block",
        )
    failed = [name for name, entry in warmed.items() if not entry.get("ok")]
    if not warmed:
        add("Engine starts", "pending", "not started", no_start)
    elif failed:
        memory = any(
            "Free memory on device" in line
            for name in failed
            for line in warmed[name].get("log", [])
        )
        add(
            "Engine starts",
            "refused",
            f"failed: {', '.join(failed)}",
            "too little free GPU memory for R2's budgets (vLLM's check)"
            if memory
            else "a start failed: warm.json keeps its log lines and error",
        )
    else:
        add("Engine starts", "ok", ", ".join(f"{n} {e['ready_s']} s" for n, e in warmed.items()))

    # The GPU's own state.
    lent = lent_gpu(host)
    if card.get("count", 1) > 1 or card.get("wanted"):
        listed = "; ".join(card.get("all") or [])
        if lent:
            add("GPU", "ok", f"GPU {card.get('index')} of {card.get('count')}: {lent}")
        else:
            add(
                "GPU",
                "refused",
                f"FLATSHARE_GPU_ID={card['wanted']} names none of: {listed}"
                if card.get("wanted")
                else f"{card.get('count')} GPUs, none chosen: {listed}",
                "the core runs on one GPU, which the units, gpu-guard and the sampler must name: "
                "FLATSHARE_GPU_ID=INDEX|UUID bootstrap.sh stage (env.sh keeps it)",
            )
    elif card:
        add("GPU", "ok", f"the host's one GPU{f': {lent}' if lent else ''}")
    if gpu:
        guard = _run(
            ["bash", str(DEPLOY / "bin" / "gpu-guard")],
            env=os.environ | ({"FLATSHARE_GPU_ID": lent} if lent else {}),
        )
        processes = _gpu_processes(lent)
        add(
            "GPU exclusive",
            "ok" if guard.returncode == 0 else "refused",
            f"{len(processes or [])} compute processes",
            "" if guard.returncode == 0 else guard.stderr.strip(),
        )
    if card:
        mode = card.get("compute_mode")
        if mode is None:
            add(
                "Compute mode",
                "pending",
                "not read",
                "bootstrap.sh gpu reads it with the card's other facts",
            )
        elif mode.lower() == "default":
            add("Compute mode", "ok", mode)
        else:
            add(
                "Compute mode",
                "refused",
                mode,
                "the STT, the TTS and the engine each hold a CUDA context, which only the Default mode allows at once "
                f"(Exclusive_Process: one at a time; warm's first start would fail): nvidia-smi -i {lent or 0} -c "
                "DEFAULT (root), or the mps variant, whose one MPS server holds the context (not the core's "
                "configuration)",
            )
    mig = card.get("mig")
    if mig and mig.lower() == "enabled":
        # A slice is one GPU to the host, but the profile's budgets and the forecast are the whole card's.
        slices = card.get("mig_devices") or []
        off = "turn MIG off (nvidia-smi -mig 0, root and a GPU reset)"
        if len(slices) == 1:
            allowed = os.environ.get("FLATSHARE_ALLOW_MIG") == "1"
            add(
                "MIG",
                "degraded" if allowed else "refused",
                f"enabled, one slice ({slices[0]['profile']}, {slices[0]['uuid']})",
                "FLATSHARE_ALLOW_MIG=1: the core on one slice of the card"
                if allowed
                else "a MIG slice: the profile's memory budgets and the forecast are the whole card's; "
                f"{off}, or measure on the slice anyway: FLATSHARE_ALLOW_MIG=1",
                "a slice is a smaller card: the forecast does not hold" if allowed else "",
            )
        else:
            add(
                "MIG",
                "refused",
                f"enabled, {len(slices)} slices",
                f"MIG is on: the core runs on the whole card; {off}",
            )
    elif card:
        add("MIG", "ok", str(mig))
    power = card.get("power_limit_w")
    if card:
        add(
            "Power cap, link and ECC",
            "ok",
            f"{power} W, PCIe gen {card.get('pcie_gen')} x{card.get('pcie_width')}, ECC {card.get('ecc')}",
            "",
            "recorded; campaign.sh samples clocks, power and throttle reasons at 1 Hz",
        )

    # The host's services.
    if supervisor(host) == "systemd" and not manager_usable(host):
        add(
            "systemd --user and linger",
            "refused",
            f"env.sh keeps systemd; systemd --user {bool(host.get('systemd_user'))}, linger {host.get('linger')}",
            "the user manager stops the units with the last session: enable lingering (loginctl enable-linger), or "
            "move the units to deploy/bin/flatshare-svc: FLATSHARE_SUPERVISOR=auto bootstrap.sh stage",
        )
    elif supervisor(host) == "systemd":
        add("systemd --user and linger", "ok", "systemd --user, lingering")
    else:
        add(
            "systemd --user and linger",
            "degraded",
            f"systemd --user {bool(host.get('systemd_user'))}, linger {host.get('linger')}"
            + ("; FLATSHARE_SUPERVISOR=flatshare-svc" if overridden() else ""),
            "supervisor instead of systemd (deploy/bin/flatshare-svc with the systemctl and journalctl shims)",
            "none on the measurements",
        )
    run = runtime_dir(host)
    longest = len(str(run / ZMQ_SOCKET).encode())
    if longest > SOCKET_PATH:
        add(
            "Socket paths",
            "refused",
            f"%t is {run}: vLLM's ZMQ sockets would take {longest} bytes, past {SOCKET_PATH}",
            "the engine would fail at its first start: a shorter runtime directory (XDG_RUNTIME_DIR, mode 0700; "
            "under flatshare-svc, XDG_STATE_HOME/flatshare-svc/run where the session has none), "
            "then bootstrap.sh stage",
        )
    else:
        add(
            "Socket paths",
            "ok",
            f"%t is {run}: vLLM's ZMQ sockets take {longest} of {SOCKET_PATH} bytes",
        )
    other, split = active_elsewhere(host)
    if split:
        add(
            "One supervisor",
            "refused",
            f"{supervisor(host)} runs the units, and {other} runs {', '.join(split)}",
            f"a split control plane, an earlier stage's: stop them under {other}, then bootstrap.sh stage again "
            "(bootstrap.sh moves the units itself on FLATSHARE_SUPERVISOR=auto or flatshare-svc)",
        )
    else:
        add("One supervisor", "ok", f"{supervisor(host)}, none of the units under {other}")
    if host.get("userns") is True:
        add(
            "User namespaces",
            "ok",
            "the voice plane in a network namespace of its own, and each GPU server in one of its own",
        )
    else:
        # The voice plane in the host's namespace, under either supervisor: smoke.sh's listener audit
        # (deploy/bin/audit-listeners) accepts what it holds there by the FLATSHARE_VOICENET=host that in-voicenet marks
        # its processes with, and refuses anything else of ours, as it does in the namespaced path.
        public = _public_addresses()
        add(
            "User namespaces",
            "degraded",
            "refused" if host.get("userns") is False else "not checked (no unshare or ip)",
            "voice plane in the host's network namespace"
            + (
                f"; the host's public addresses {', '.join(public)} must drop inbound UDP"
                if public
                else ""
            )
            + "; smoke.sh's listener audit accepts its UDP sockets, loopback TCP listeners and file sockets there "
            "(FLATSHARE_VOICENET=host) and nothing else of ours; the GPU servers run there too, by their drop-ins' "
            "FLATSHARE_PRIVATE_NET=host (deploy/bin/in-private-net), where the audit accepts their "
            "engine cores' loopback "
            "TCP listeners on ephemeral ports and CUDA's abstract sockets, which any user of the host reaches, and "
            "nothing else of theirs; allowing unprivileged user namespaces gives "
            "each its own (Ubuntu 24.04: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0, then "
            "bootstrap.sh stage)",
            "none on latency; isolation differs",
        )
    # smoke.sh (block 0) audits the listening sockets with ss on the GPU node.
    add(
        "Listener audit",
        "ok" if shutil.which("ss") else "refused",
        "ss" if shutil.which("ss") else "no ss",
        ""
        if shutil.which("ss")
        else "smoke.sh's listener audit needs ss (iproute2): bootstrap.sh stage installs it "
        "where sudo and apt work",
    )
    if few := [w for w in node.node.get("warnings", []) if "logical CPUs" in w]:
        add(
            "Cores",
            "degraded",
            f"{sum(len(c) for c in host.get('cpus', {}).values())} CPUs",
            few[0],
            "callers' lateness and AIPerf's timing are recorded per point and shown",
        )
    elif node.node.get("plan"):
        plan = node.plan
        add("Cores", "ok", ", ".join(f"{k} {plan[k]}" for k, _ in profile.CORES))
    if host.get("peer"):
        rtt = host["peer"].get("rtt_ms")
        add(
            "RTT to the control host (split mode)",
            "ok" if rtt and sorted(rtt)[len(rtt) // 2] <= 5 and max(rtt) <= 20 else "refused",
            f"{rtt}",
            ""
            if rtt and sorted(rtt)[len(rtt) // 2] <= 5 and max(rtt) <= 20
            else "p50 above 5 ms or p99 above 20 ms: single-host mode instead",
        )

    # The control plane, which bootstrap.sh stage starts.
    down = [
        unit
        for unit in CONTROL
        if not shutil.which("systemctl")
        or _run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode != 0
    ]
    if len(down) == len(CONTROL):
        add("Control plane", "pending", "not started", "bootstrap.sh stage starts it")
    elif down or _get(gateway_url, "/health") is None:
        add(
            "Control plane",
            "refused",
            f"down: {', '.join(down) or 'the gateway does not answer /health'}",
            "journalctl --user -u UNIT says why",
        )
    else:
        add("Control plane", "ok", f"{len(CONTROL)} units active, the gateway answers /health")

    # What the campaign reads from disk.
    missing = []
    for config in ("engine.yaml", "stt.yaml"):
        pinned = _yaml_model(node.dir / config)
        if pinned is None or not _hf_snapshot(hub_cache(host, home), *pinned):
            missing.append(f"{pinned[0]}@{pinned[1][:7]}" if pinned else config)
    add(
        "Models",
        "refused" if missing else "ok",
        ", ".join(missing) if missing else "the engine's and the STT's at their revisions",
        f"missing from {hub_cache(host, home)}: bootstrap.sh stage downloads them"
        if missing
        else "",
    )
    checks.append(data_check(root, data))
    binaries = []
    for name, args in (("flatshare-gateway", ["--version"]), ("livekit-server", ["--version"])):
        binary = root / "bin" / name
        try:
            ok = _run([str(binary), *args], timeout=30).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if not ok:
            binaries.append(name)
    add(
        "Binaries",
        "refused" if binaries else "ok",
        ", ".join(binaries) if binaries else "flatshare-gateway, livekit-server",
        "missing or not runnable in ~/flatshare/bin: bootstrap.sh stage" if binaries else "",
    )
    disk = _free(root)
    if disk is not None:
        status = "ok" if disk >= 20 * GIB else ("degraded" if disk >= 5 * GIB else "refused")
        add(
            "Disk",
            status,
            f"{disk / GIB:.0f} GiB free under ~/flatshare",
            "" if status == "ok" else "little room for the points' records [estimate: 20 GiB]",
        )
    return checks


def write_preflight(node: Node, checks: list[Check], out: Path | None = None) -> Path:
    out = out or node.dir / "preflight.json"
    counts = {status: sum(c.status == status for c in checks) for status in STATUSES}
    _write(
        out,
        json.dumps(
            {
                "time": time.time(),
                "host": node.host.get("hostname"),
                "counts": counts,
                "checks": [asdict(c) for c in checks],
            },
            indent=2,
        )
        + "\n",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="host.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    node_help = "the node's directory (NODE), as bootstrap.sh makes it"
    for name, text in (
        (
            "env",
            "write NODE/env.sh, with the supervisor's shims where flatshare-svc runs the units",
        ),
        (
            "supervisor",
            "print what runs the units: flatshare-svc or systemd (FLATSHARE_SUPERVISOR and the facts)",
        ),
        ("units", "link the units and this host's drop-ins"),
        ("config", "write the control plane's configuration"),
    ):
        commands.add_parser(name, help=text).add_argument("node", type=Path, help=node_help)
    p = commands.add_parser(
        "jit-check", help="build FlashInfer's sampler as the engine's first start would"
    )
    p.add_argument("node", type=Path, help=node_help)
    p.add_argument(
        "--arch",
        help="the compute capability to build for (default: the card's, else the profile's)",
    )
    p.add_argument(
        "--if-stale",
        action="store_true",
        help="build only where the last build failed or was for another compiler or card",
    )
    p = commands.add_parser("preflight", help="check the host for the campaign")
    p.add_argument("node", type=Path, help=node_help)
    p.add_argument("--gpu", action="store_true", help="also run the checks that need the card")
    p.add_argument("--out", type=Path, help="where to write preflight.json (default: in NODE)")
    p = commands.add_parser("warm", help="start the engine once per compile configuration")
    p.add_argument("node", type=Path, help=node_help)
    p.add_argument(
        "--configs", type=Path, help="NAME VARIANT [FLAG...] lines instead of the core's three"
    )
    p = commands.add_parser(
        "warmed", help="whether every core compile configuration started (exit status 1 if not)"
    )
    p.add_argument("node", type=Path, help=node_help)
    p = commands.add_parser("headers", help="whether PYTHON has Python.h (exit status 1 if not)")
    p.add_argument("python", type=Path, help="the Python interpreter to look in")
    p = commands.add_parser(
        "with-engine-env", help="run a command with the engine unit's environment"
    )
    p.add_argument("node", type=Path, help=node_help)
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="the command and its arguments")
    args = parser.parse_args(argv)
    if args.command == "headers":
        found = python_headers(args.python)
        print(
            f"host: {args.python}: "
            + (
                f"Python.h in {found[1]}"
                if found and found[0]
                else f"no Python.h ({found[1] if found else 'no such interpreter'})"
            )
        )
        return 0 if found and found[0] else 1
    if args.command == "with-engine-env":
        if not args.cmd:
            parser.error("with-engine-env needs a command")
        env = os.environ | unit_environment(
            args.node / "systemd" / "flatshare-engine.service", _home()
        )
        os.execvpe(args.cmd[0], args.cmd, env)
    try:
        node = Node.load(args.node)
        if args.command == "env":
            print(f"host: wrote {write_env(node)} ({supervisor(node.host)} runs the units)")
        elif args.command == "supervisor":
            print(supervisor(node.host))
        elif args.command == "warmed":
            done = (_json(_home() / "flatshare" / "cache" / "warm.json") or {}).get(
                "configurations", {}
            )
            cold = [name for name, *_ in WARM if not (done.get(name) or {}).get("ok")]
            print(
                f"host: {'cold: ' + ', '.join(cold) if cold else 'every core compile configuration is warm'}"
            )
            return 1 if cold else 0
        elif args.command == "units":
            for link in install_units(node):
                print(f"host: linked {link}")
        elif args.command == "config":
            for path in write_config(node):
                print(f"host: wrote {path}")
        elif args.command == "jit-check":
            if args.if_stale and not args.arch:
                recorded = _json(_home() / "flatshare" / "cache" / "jit-check.json") or {}
                if recorded.get("ok") and jit_current(node)[0]:
                    print(
                        f"host: FlashInfer's sampler is built for {recorded['arch']} with {recorded['cuda_home']}"
                    )
                    return 0
            result = jit_check(node, cap=args.arch)
            if not result["ok"]:
                print(
                    f"host: FlashInfer's sampler did not build for {result['arch']}:\n{result['error']}",
                    file=sys.stderr,
                )
                return 1
            print(
                f"host: built FlashInfer's sampler for {result['arch']} in {result['seconds']} s: {result['library']}"
            )
        elif args.command == "preflight":
            checks = preflight(node, gpu=args.gpu)
            out = write_preflight(node, checks, args.out)
            width = max(len(c.check) for c in checks)
            for c in checks:
                print(
                    f"{c.check:<{width}}  {c.status:<8}  {c.value}"
                    + (f"\n{'':<{width}}  {'':<8}  {c.reason}" if c.reason else "")
                )
            refused = [c.check for c in checks if c.status == "refused"]
            print(
                f"host: {out}: "
                + ", ".join(f"{sum(c.status == s for c in checks)} {s}" for s in STATUSES)
            )
            return 1 if refused else 0
        else:
            record = warm(node, configurations(args.configs))
            return 0 if all(entry["ok"] for entry in record["configurations"].values()) else 1
    except HostError as e:
        print(f"host: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
