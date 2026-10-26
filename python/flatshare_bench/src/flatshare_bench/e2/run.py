"""``flatshare-bench e2 run`` measures one E2 point on cyan, the control host, from its arm file:

1. it records the engine behind the tunnel from the engine's own ``/server_info`` and checks the
   arm's ``[engine]`` against it and the GPU node's hardware profile (the frozen box's, unless the
   host's settings or ``--profile`` name another);
2. it makes cyan's units serve the point: the router (SMG, for arm 8b), the gateway's configuration
   (``gateway.toml`` with the arm's ``[gateway]`` table merged in) and the voice agent's preemptive
   generation and lifecycle events, restarting only the units whose configuration changes;
3. it empties the engine's prefix cache, so that every point starts from the same state, and, with
   agent sessions, starts AIPerf's ``agentic_replay`` (``configs/agentic_replay.yaml``) on the
   load-generator cores and waits for its profiling phase and a warm-up;
4. it snapshots the gateway's and the engine's ``/metrics``, runs ``flatshare-voice bots`` inside the
   voice-plane namespace (``--save-audio`` has them save each call's audio, for the demo video) and
   snapshots again: the calls are the measurement window, through which an agent point samples the
   engine's KV use and requests every half second (``--kv-sample-s``; 0 samples nothing, as W5's points
   ran: the sampler came after them), on one connection to the engine that it keeps for the window;
5. it stops AIPerf, keeps the gateway's request log of the calls and the voice worker's journal of
   them (its turn lines and its sessions' event counts), and writes ``summary.json``.

The run reaches the engine by its address (:mod:`flatshare_bench.transport`): on cyan, the tunnel's
socket. Its engine check and cache resets need vLLM's development routes, which only the measurement
drop-in serves (``deploy/phastform/variant.sh measure on``, which ``benchmarks/e2/campaign.sh`` runs).

The GPU node's engine is switched by ``benchmarks/e2/campaign.sh`` (``variant.sh use``), never here.
A failure ends the point: ``run.json`` keeps ``"status": "failed"`` and the error, and every process
the run started is stopped by its process group. ``flatshare-bench e2 report`` (:mod:`.report`) builds the
frontier from the run directories.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import orjson

from flatshare_bench.e2.arm import Arm, E2Error, Engine, Voice, flag_values, load_arm
from flatshare_bench.e2.record import (
    OUTCOMES,
    gateway_request_lines,
    kv_values,
    logged_requests,
    model_id,
    render,
    summarize_point,
    worker_lines,
)
from flatshare_bench.kv_metrics import fetch
from flatshare_bench.tiers import pool_tokens
from flatshare_bench.transport import client, runtime_socket
from flatshare_stats import metric_samples

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
"""The AIPerf configuration, from the source tree (``uv sync`` installs the package editable)."""
VARIANTS = Path(__file__).resolve().parents[5] / "deploy" / "phastform" / "variants"
"""The GPU node's engine variants, from the same source tree: what each adds to the engine's command
line, which the engine check expects."""
PROFILES = Path(__file__).resolve().parents[5] / "deploy" / "profiles"
"""The GPU nodes' hardware profiles (deploy/profile.py renders a node's deployment from one), from the
same source tree: the model and the scheduler settings a node's engine.yaml gives its engine, which
the engine check expects."""
BOX_PROFILE = "rtx-5060-ti"
"""The frozen box's profile: Phastform's, and Gray's."""
HOST_FILE = Path(".config") / "flatshare" / "e2-host.toml"
"""Where a control host other than cyan sets :class:`Host`'s fields, under its home directory: a
single host its cores from its profile's core plan, and its profile. cyan has none."""

PROFILING = "Credit phase start: profiling"
"""AIPerf's log line when its measured phase begins, after the dataset is built and warmed up."""
PROFILING_TIMEOUT_S = 1200.0
"""Building the full corpus's dataset takes about 4 minutes on cyan; configs/aiperf.env allows 900 s."""
AIPERF_DURATION_S = 10800
"""AIPerf's duration: the replay is stopped when the calls end, so this only has to outlast them."""
AIPERF_STOP_S = 120.0
"""How long AIPerf may take to cancel its requests and write its exports after SIGINT."""
CALL_TURN_S = 130.0
"""The longest a call's turn can take: the line, a reply, a barge-in and a reply, the quiet wait."""
CALLERS_EXIT_S = 120.0
"""The longest the callers may take to exit once their calls ended: with --save-audio they write each
call's audio then."""
POLL_S = 1.0
"""How often a run checks on the tools and services it waits for."""
SESSIONS_END_S = 30.0
"""How long the calls' sessions may take to log their end once the callers have left: a session ends
when its room is deleted, and its job's shutdown, which sends the last events, may take 10 s."""

SMG = "flatshare-smg"
GATEWAY = "flatshare-gateway"
WORKER = "flatshare-voice-worker"
E2_GATEWAY = "e2-gateway"
NO_PREEMPTIVE = "no-preemptive-generation"
NO_EVENTS = "no-events"
"""Drop-ins under deploy/cyan/variants/ that e2 run links into cyan's units as ``variant-NAME.conf``."""
GATEWAY_IDLE_S = 30.0
"""How long the gateway may take to end its requests once a point's tools have stopped."""
BUSY = ("flatshare_gateway_streams_active", "flatshare_gateway_admission_waiting")
"""The gateway's gauges of the responses it is relaying and of the requests waiting at its admission
point: it is idle when both are at zero (:func:`_ended`)."""
KV_SAMPLE_S = 0.5
"""How often an agent point samples the engine's KV use and its running and waiting requests from its
``/metrics`` while the calls run (P-1's sampler; record.KV_GAUGES), from which the summary takes the
measured running share; ``flatshare-bench e2 run --kv-sample-s`` sets another, 0 none (W5's points ran without
it)."""

SGLANG = "sglang"
"""The one engine variant that serves SGLang; every other variant serves vLLM."""


# cyan


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
        names = [f.name for f in fields(cls) if f.name != "home"]
        if unknown := sorted(set(table) - set(names)):
            raise E2Error(f"{path} sets {', '.join(unknown)}; a host sets {', '.join(names)}")
        if wrong := sorted(key for key, value in table.items() if not isinstance(value, str)):
            raise E2Error(f"{path}: {', '.join(wrong)} must be strings")
        return cls(home=home, **table)

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


def configure_cyan(arm: Arm, host: Host, systemd: Systemd) -> dict[str, Any]:
    """Make cyan's units serve ``arm``, restarting only those whose configuration changes; returns
    what they run, for run.json."""
    env = systemd.environment(WORKER)
    if env.get("FLATSHARE_TURN_DETECTION") != "vad":
        raise E2Error(
            f"{WORKER} ends turns with LiveKit's end-of-turn model; E2 points need fixed endpointing: "
            "link deploy/cyan/variants/vad-turns (ops/RUNBOOK.md)"
        )
    applied = _apply(host, systemd, arm.router, arm.gateway, arm.voice)
    return applied | {"llm_path": env.get("FLATSHARE_LLM_UDS")}


def reset_cyan(host: Host, systemd: Systemd) -> dict[str, Any]:
    """Undo what e2 run changes on cyan: SMG stopped, the gateway back on gateway.toml, and preemptive
    generation and lifecycle events on."""
    with exclusive(host):
        return _apply(host, systemd, None, {}, Voice())


def _apply(
    host: Host,
    systemd: Systemd,
    router: str | None,
    overlay: Mapping[str, Any],
    voice: Voice,
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
        _until(
            lambda: "registered worker" in systemd.journal(WORKER, since),
            90.0,
            f"{WORKER} did not register with LiveKit within 90 s (journalctl --user -u {WORKER})",
        )
    return {
        "router": router,
        "gateway": {"restarted": gateway, "config": _redacted(config)},
        "worker": {
            "restarted": worker,
            "preemptive_generation": voice.preemptive_generation,
            "events": voice.events,
        },
    }


GATEWAY_HEADER = (
    "# Generated by flatshare-bench e2 run: gateway.toml with an arm file's [gateway] table merged in.\n"
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


def _write_private(path: Path, text: str) -> bool:
    """Write ``text`` to a file only its owner can read (it holds key digests); returns whether it
    changed."""
    if path.exists() and path.read_text() == text:
        return False
    staged = path.with_name(path.name + ".new")
    staged.write_text(text)
    staged.chmod(0o600)
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


def _redacted(config: Any) -> Any:
    """A configuration (the gateway's, an engine's) without its credentials."""
    if isinstance(config, Mapping):
        return {
            k: (["<redacted>"] * len(v) if isinstance(v, list) else "<redacted>")
            if k in SECRETS and v
            else _redacted(v)
            for k, v in config.items()
        }
    if isinstance(config, list):
        return [_redacted(v) for v in config]
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


# The engine


@dataclass(frozen=True)
class EngineInfo:
    """The engine's own account of itself, from its ``/server_info``."""

    server: str
    """``vllm`` or ``sglang``."""
    version: str | None
    pool_tokens: int | None
    """The GPU KV pool in tokens."""
    settings: Mapping[str, Any]
    """Settings by option name: vLLM's scheduler and cache configuration, SGLang's server arguments."""
    config: Mapping[str, Any]
    """The whole configuration, for engine.json."""


SETTING_NAMES = {"--scheduling-policy": "policy"}
"""vLLM flags whose setting has another name; any other ``--a-flag`` sets ``a_flag``."""

FROZEN = (
    "--scheduling-policy", "priority",
    "--max-num-batched-tokens", "512",
    "--long-prefill-token-threshold", "256",
)  # fmt: skip
"""The scheduler settings of deploy/phastform/engine.yaml that E2's arms change (a test keeps the two
equal). A vLLM point expects them wherever its flags say nothing else, so it cannot run on what an
earlier point left on the box."""


MEASURE = (
    "vLLM serves /server_info and the cache resets only in its development mode, which the measurement "
    "drop-in sets: deploy/phastform/variant.sh measure on"
)


def load_profile(name: str, profiles: Path = PROFILES) -> dict[str, Any]:
    """A hardware profile, its base's values under its own where its [profile] table names a base, as
    deploy/profile.py loads it."""
    path = profiles / f"{name}.toml"
    if not path.is_file():
        raise E2Error(f"no hardware profile {name} in {profiles}")
    profile = tomllib.loads(path.read_text())
    base = profile.get("profile", {}).pop("base", None)
    if base is None:
        return profile
    merged = load_profile(base, profiles)
    for section, table in profile.items():
        merged[section] = merged.get(section, {}) | table if isinstance(table, dict) else table
    return merged


def profile_flags(name: str, profiles: Path = PROFILES) -> tuple[str, ...]:
    """The flags a GPU node's engine.yaml amounts to, by its hardware profile, among those the engine
    reports: FROZEN's scheduler settings as the profile sets them, and its model, revision and context.
    A vLLM point expects them wherever its flags say nothing else."""
    profile = load_profile(name, profiles)
    model, engine = profile["model"], profile["engine"]
    return (
        "--scheduling-policy", "priority",  # engine.yaml's for every profile (deploy/templates)
        "--max-num-batched-tokens", str(engine["max_num_batched_tokens"]),
        "--long-prefill-token-threshold", str(engine["long_prefill_token_threshold"]),
        "--model", model["name"],
        "--revision", model["revision"],
        "--max-model-len", str(model["max_model_len"]),
    )  # fmt: skip


def describe_engine(engine: str) -> EngineInfo:
    """The engine at the address ``engine``: vLLM (whose ``/server_info`` needs its development mode, from
    the measurement drop-in) or SGLang. A vLLM engine's settings are its cache, scheduler and kernel
    configuration's, and its model, revision and context."""
    info = _get_json(engine, "/server_info", params={"config_format": "json"}, missing=MEASURE)
    if "vllm_config" in info:
        config = info["vllm_config"]
        model = config.get("model_config") or {}
        name, revision = model_id(model)
        settings = {
            **config.get("cache_config", {}),
            **config.get("scheduler_config", {}),
            **(config.get("kernel_config") or {}),
            **{
                key: value
                for key, value in (
                    ("model", name),
                    ("revision", revision),
                    ("max_model_len", model.get("max_model_len")),
                )
                if value is not None
            },
        }
        version = _get_json(engine, "/version").get("version")
        return EngineInfo("vllm", version, pool_tokens(fetch(engine)), settings, config)
    if "internal_states" in info:
        config = {k: v for k, v in info.items() if k != "internal_states"}
        return EngineInfo(
            SGLANG, info.get("version"), info.get("max_total_num_tokens"), config, config
        )
    raise E2Error(f"{engine}: /server_info is neither vLLM's nor SGLang's")


SCHEDULER = "--scheduler-cls"
ADDITIONAL = "--additional-config."
"""vLLM's flags that name the scheduler class and set one key of ``additional_config`` (make-room's
switches, ``--additional-config.make_room.reserve_tokens 0``)."""


def variant_flags(variant: str, variants: Path = VARIANTS) -> tuple[str, ...]:
    """The engine flags a variant gives its engine (frozen: none): those of the command its
    flatshare-engine.conf starts (the last ``ExecStart=``), from its first flag on, without ``--config`` (the
    frozen box's engine.yaml, whose settings FROZEN has), ``--uds`` (the engine's socket: its transport,
    not one of its settings) and ``$FLATSHARE_ENGINE_FLAGS``, where ``variant.sh use`` puts the arm's flags,
    after the variant's."""
    if variant == "frozen":
        return ()
    if not (variants / variant).is_dir():
        raise E2Error(f"no engine variant {variant} in {variants}")
    conf = variants / variant / "flatshare-engine.conf"
    if not conf.exists():
        return ()  # a variant that leaves the engine's command line alone (mps)
    # systemd joins a line that ends in a backslash with the next one.
    lines = conf.read_text().replace("\\\n", " ").splitlines()
    starts = [line.partition("=")[2] for line in lines if line.startswith("ExecStart=")]
    words = shlex.split(starts[-1]) if starts else []
    first = next((i for i, word in enumerate(words) if word.startswith("--")), len(words))
    flags: list[str] = []
    for word in words[first:]:
        if flags and flags[-1] in ("--config", "--uds"):
            flags.pop()
        elif not word.startswith("$"):
            flags.append(word)
    return tuple(flags)


def check_engine(
    engine: Engine, info: EngineInfo, variants: Path = VARIANTS, base: Sequence[str] = FROZEN
) -> dict[str, list[str]]:
    """Check the arm's engine against what the engine reports. The point expects the engine's command
    line as ``variant.sh use`` makes it: on vLLM ``base``, the node's engine.yaml (the frozen box's
    scheduler settings, or :func:`profile_flags`), then the variant's flags (:func:`variant_flags`),
    then the arm's, the last of a repeated flag winning. Each flag's setting must match; on vLLM the
    scheduler class (``scheduler_cls``) must be the one
    ``--scheduler-cls`` names, vLLM's own without it, and ``additional_config`` must be exactly what the
    ``--additional-config.KEY`` flags build, so that frozen, make-room and pr54530 engines, and
    make-room with another switch, cannot pass for one another. A flag whose setting the engine does
    not report is listed as unchecked."""
    expected = SGLANG if engine.variant == SGLANG else "vllm"
    switch = f"switch it with deploy/phastform/variant.sh use {engine}"
    if info.server != expected:
        raise E2Error(f"the engine is {info.server}, the arm file needs {expected}: {switch}")
    wanted = flag_values(variant_flags(engine.variant, variants)) | flag_values(engine.flags)
    if expected == "vllm":
        wanted = flag_values(base) | wanted
    checked, unchecked = [], []
    for flag, value in wanted.items():
        if expected == "vllm" and (flag == SCHEDULER or flag.startswith(ADDITIONAL)):
            continue  # checked as a whole below
        key = SETTING_NAMES.get(flag, flag.removeprefix("--").replace("-", "_"))
        if key not in info.settings:
            unchecked.append(flag)
        elif _same(info.settings[key], value):
            checked.append(flag)
        else:
            reported = info.settings[key]
            raise E2Error(
                f"the engine runs {key} = {reported!r}, the point needs {flag} {value}: {switch}"
            )
    if expected == "vllm":
        scheduler = [SCHEDULER] if SCHEDULER in wanted else []
        if "scheduler_cls" not in info.settings:
            unchecked += scheduler
        elif info.settings["scheduler_cls"] != wanted.get(SCHEDULER):
            needs = f"{SCHEDULER} {wanted[SCHEDULER]}" if scheduler else "vLLM's own scheduler"
            raise E2Error(
                f"the engine runs scheduler_cls = {info.settings['scheduler_cls']!r}, the point "
                f"needs {needs}: {switch}"
            )
        else:
            checked += scheduler
        additional = [flag for flag in wanted if flag.startswith(ADDITIONAL)]
        if not isinstance(info.config.get("additional_config"), Mapping):
            unchecked += additional
        else:
            built = _additional_config({flag: wanted[flag] for flag in additional})
            if problem := _differs(info.config["additional_config"], built, ()):
                raise E2Error(f"{problem}: {switch}")
            checked += additional
    return {"checked": checked, "unchecked": unchecked}


def _additional_config(flags: Mapping[str, str]) -> dict[str, Any]:
    """The ``additional_config`` that ``--additional-config.KEY`` flags build: KEY's dots nest tables,
    and each value stays the flag's text."""
    config: dict[str, Any] = {}
    for flag, value in flags.items():
        *tables, key = flag.removeprefix(ADDITIONAL).split(".")
        table = config
        for name in tables:
            table = table.setdefault(name, {})
            if not isinstance(table, dict):
                raise E2Error(f"{flag} sets a key inside a value of another flag")
        if isinstance(table.get(key), dict):
            raise E2Error(f"{flag} sets a value where other flags set keys")
        table[key] = value
    return config


def _differs(reported: Mapping[str, Any], built: Mapping[str, Any], path: tuple[str, ...]) -> str:
    """Where the engine's ``additional_config`` (``reported``) is not the one the point's flags build,
    said for ``e2 run``'s error; empty when it is."""
    for key in [*built, *(key for key in reported if key not in built)]:
        name, flag = ".".join((*path, key)), ADDITIONAL + ".".join((*path, key))
        if key not in reported:
            needs = _flag_line(flag, built[key])
            return f"the engine's additional_config has no {name}, the point needs {needs}"
        runs = f"the engine runs additional_config {name} = {reported[key]!r}"
        if key not in built:
            return f"{runs}, which the point's flags do not set"
        if isinstance(built[key], dict) and isinstance(reported[key], Mapping):
            if problem := _differs(reported[key], built[key], (*path, key)):
                return problem
        elif isinstance(built[key], dict) or not _same(reported[key], built[key]):
            return f"{runs}, the point needs {_flag_line(flag, built[key])}"
    return ""


def _flag_line(flag: str, value: Any) -> str:
    """A built ``additional_config`` value as the flags that set it."""
    if not isinstance(value, dict):
        return f"{flag} {value}"
    return " ".join(_flag_line(f"{flag}.{key}", item) for key, item in value.items())


def _same(reported: Any, text: str) -> bool:
    if isinstance(reported, bool):
        return text.lower() == str(reported).lower()
    if isinstance(reported, int | float):
        try:
            return float(text) == float(reported)
        except ValueError:
            return False
    return str(reported) == text


def reset_prefix_cache(engine: str, server: str, timeout: float = 120.0) -> None:
    """Empty the prefix cache (and vLLM's CPU tier) of the engine at ``engine``. Both engines refuse
    while requests run or KV transfers are in flight, so this retries."""
    path = "/reset_prefix_cache?reset_external=true" if server == "vllm" else "/flush_cache"

    def done() -> bool:
        with client(engine, timeout=30) as http:
            response = http.post(path)
        if response.status_code == 404:
            raise E2Error(f"{engine}: no {path.split('?')[0]}. {MEASURE}")
        if server == "vllm":
            return response.status_code == 200 and response.json().get("success") is True
        return response.status_code == 200

    _until(done, timeout, f"the engine did not reset its prefix cache within {timeout:.0f} s")


def _get_json(address: str, path: str, missing: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """The JSON of ``GET path`` from the server at ``address``; ``missing`` says why a 404 may come."""
    try:
        with client(address, timeout=30) as http:
            response = http.get(path, **kwargs)
        if response.status_code == 404 and missing:
            raise E2Error(f"{address}: no {path}. {missing}")
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise E2Error(f"GET {path} from {address}: {exc}") from None


# Processes


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


def _until(done: Any, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while not done():
        if time.monotonic() > deadline:
            raise E2Error(message)
        time.sleep(POLL_S)


# A point


def run_point(
    arm_file: Path,
    out: Path,
    host: Host | None = None,
    systemd: Systemd | None = None,
    *,
    save_audio: bool = False,
    gpu_source: Mapping[str, str | None] | None = None,
    agents_seed: int | None = None,
    kv_sample_s: float | None = KV_SAMPLE_S,
) -> dict[str, Any]:
    """Measure the point of ``arm_file`` into the directory ``out``; returns the summary. A
    directory with a complete point is never overwritten, and an earlier attempt that did not
    complete is kept beside it. ``save_audio``, a run option rather than the arm's, has the callers
    save each call's audio (``flatshare-voice bots --save-audio``). ``gpu_source`` is each GPU node's
    deployed revision (its ~/flatshare/src/DEPLOYED, None where it has none), which campaign.sh
    reads over its SSH at the point's start and run.json keeps beside cyan's; None when unknown.
    ``agents_seed``, a campaign's (``agents-seed`` in its file), replaces the arm file's ``[agents]
    seed`` for AIPerf; run.json keeps the seed the replay ran with (``agents_seed``). An agent point
    samples the engine's KV every ``kv_sample_s`` while the calls run; 0 or None samples nothing (W5's
    points ran without the sampler, and tests whose fake engines count their /metrics reads), and
    run.json keeps the interval the point sampled at (``kv_sample_s``, null when it did not)."""
    if not hasattr(os, "waitid"):
        raise E2Error(NEEDS_WAITID)
    if stale := _stale_install():
        raise E2Error(stale)
    if agents_seed is not None and agents_seed < 0:
        raise E2Error(f"the agents' seed cannot be negative: {agents_seed}")
    if kv_sample_s is not None and kv_sample_s < 0:
        raise E2Error(f"the KV sampler's interval cannot be negative: {kv_sample_s}")
    kv_sample_s = kv_sample_s or None  # 0 samples nothing
    arm = load_arm(arm_file)
    seed = arm.agents.seed if agents_seed is None else agents_seed
    host = host or Host.load()
    systemd = systemd or Systemd()
    with exclusive(host):
        _fresh(out)
        shutil.copyfile(arm_file, out / "arm.toml")
        record = _Record(out, arm, host, gpu_source)
        record["save_audio"] = save_audio
        # The seed AIPerf replays with: the campaign's when it set one, else the arm file's.
        record["agents_seed"] = seed if arm.agents.sessions else None
        # The KV sampler's interval, null where the point did not sample (voice alone, or 0).
        record["kv_sample_s"] = kv_sample_s if arm.agents.sessions else None
        try:
            # Under the lock no other run's tools are alive: a group named in its scope, or a
            # process marked in it, is left from a run that was killed before it could stop them.
            if leftovers := kill_left(tool_scope(host)):
                record["leftovers_killed"] = leftovers
                record.say(f"killed {len(leftovers)} processes left by an earlier run: {leftovers}")
            with contextlib.ExitStack() as running:
                summary = _measure(
                    arm, out, host, systemd, record, running, save_audio, seed, kv_sample_s
                )
        except BaseException as exc:
            record.fail(exc)
            raise
        record.finish()
        return summary


def _measure(
    arm: Arm,
    out: Path,
    host: Host,
    systemd: Systemd,
    record: _Record,
    running: contextlib.ExitStack,
    save_audio: bool,
    seed: int,
    kv_sample_s: float | None,
) -> dict[str, Any]:
    base = profile_flags(host.profile)
    record["profile"] = host.profile
    if host.preflight:
        record["preflight"] = preflight_summary(Path(host.preflight))
    server = _check_engine(arm, host.engine_url, out, record, base)
    record.say(f"configuring cyan for {arm.name}")
    record["cyan"] = configure_cyan(arm, host, systemd)
    _until(
        lambda: _ready(host.gateway_url),
        60.0,
        f"{host.gateway_url}/ready did not answer within 60 s: the engine or the router is down",
    )
    reset_prefix_cache(host.engine_url, server)
    # Before the replay and the calls the gateway is idle: the base of the check that its journal kept a
    # line for every request it ended.
    idle = (time.time(), _ended(host))
    # The model the replay asks for: the engine's, as the node's profile and the arm's flags start it.
    model = (flag_values(base) | flag_values(arm.engine.flags))["--model"]
    aiperf = (
        _start_aiperf(arm, host, out, record, running, seed, model) if arm.agents.sessions else None
    )
    start = _snapshot(host, out, "start")
    sampler = None
    if aiperf is not None and kv_sample_s is not None:
        sampler = _KvSampler(host.engine_url, out / "metrics", kv_sample_s)
        running.callback(sampler.stop)
    end, bots = _call(arm, host, out, record, running, aiperf, save_audio)
    if sampler is not None:
        sampler.stop()
    _snapshot(host, out, "end")
    _callers_exit(bots)
    record["window"] = {"start": start, "end": end}
    if aiperf is not None:
        record.say("stopping AIPerf")
        record["aiperf"] = {"exit_status": aiperf.stop(AIPERF_STOP_S)}
        if aiperf.status:
            raise aiperf.failure("failed when stopped")
    _gateway_log(systemd, out, start, end)
    record["gateway_log"] = _check_gateway_log(host, systemd, idle)
    _worker_journal(systemd, out, start, arm.voice.callers)
    try:
        metrics = summarize_point(
            out,
            start,
            end,
            agents=bool(arm.agents.sessions),
            warmup_turns=arm.voice.warmup_turns,
        )
    except ValueError as exc:
        raise E2Error(str(exc)) from None
    summary = {"name": arm.name, "arm": arm.arm, "label": arm.label} | metrics
    _write_json(out / "summary.json", summary)
    record.say(render(summary))
    return summary


def _check_engine(
    arm: Arm, url: str, out: Path, record: _Record, base: Sequence[str] = FROZEN
) -> str:
    """Record the engine (``engine.json`` and run.json's ``engine``) and check it against the arm's;
    returns its server."""
    info = describe_engine(url)
    _write_json(out / "engine.json", _redacted(info.config))
    engine = {"server": info.server, "version": info.version, "pool_tokens": info.pool_tokens}
    record["engine"] = engine  # recorded before the check, which fails on a wrong engine
    record["engine"] = engine | {"flags": check_engine(arm.engine, info, base=base)}
    return info.server


def _start_aiperf(
    arm: Arm,
    host: Host,
    out: Path,
    record: _Record,
    running: contextlib.ExitStack,
    seed: int,
    model: str,
) -> Process:
    """Start the replay with ``seed``, asking for ``model``, and return once it has been profiling for
    the warm-up."""
    agents = arm.agents
    record.say(f"AIPerf: {agents.sessions} session trees through the gateway, seed {seed}")
    env = _aiperf_env(arm, host, out, seed, model)
    aiperf = Process("AIPerf", host.aiperf(), out / "aiperf.log", env, tool_scope(host))
    running.callback(aiperf.stop, AIPERF_STOP_S)
    deadline = time.monotonic() + PROFILING_TIMEOUT_S
    while PROFILING not in aiperf.log.read_text(errors="replace"):
        if aiperf.exited():
            raise aiperf.failure("exited before its profiling phase")
        if time.monotonic() > deadline:
            raise aiperf.failure(f"did not start profiling within {PROFILING_TIMEOUT_S:.0f} s")
        time.sleep(POLL_S)
    record.say(f"AIPerf is profiling; {agents.warmup:g} s of warm-up")
    deadline = time.monotonic() + agents.warmup
    while (left := deadline - time.monotonic()) > 0:
        if aiperf.exited():
            raise aiperf.failure("exited during the warm-up")
        time.sleep(min(POLL_S, left))
    return aiperf


def _call(
    arm: Arm,
    host: Host,
    out: Path,
    record: _Record,
    running: contextlib.ExitStack,
    aiperf: Process | None,
    save_audio: bool,
) -> tuple[float, Process]:
    """Run the calls; returns when they ended, and the callers' process, which may still be writing
    the calls' audio (:func:`_callers_exit`). The replay must outlast the calls."""
    voice = arm.voice
    warmup = f" after {voice.warmup_turns} warm-up" if voice.warmup_turns else ""
    record.say(f"calls: {voice.callers} callers, {voice.turns} turns each{warmup}")
    options = {
        "--corpus": host.root / "data" / "callers",
        "--callers": voice.callers,
        "--turns": voice.turns,
        "--barge-in": voice.barge_in,
        "--seed": voice.seed,
        # Only when there are some, as the callers before the option took no such flag.
        **({"--warmup-turns": voice.warmup_turns} if voice.warmup_turns else {}),
        "--out": out / "bots",
    }
    command = [*host.bots(), *(str(word) for option in options.items() for word in option)]
    if save_audio:
        command.append("--save-audio")
    env = os.environ | read_env(host.config / "livekit.env")
    bots = Process("flatshare-voice bots", command, out / "bots.log", env, tool_scope(host))
    running.callback(bots.stop, 30.0)
    deadline = time.monotonic() + CALL_TURN_S * (voice.warmup_turns + voice.turns) + 120
    # The callers write their summary as their calls end, before any call's audio.
    ended = out / "bots" / "summary.json"
    while not bots.exited() and not ended.exists():
        if aiperf is not None and aiperf.exited():
            raise aiperf.failure("exited while the calls ran, so its load did not cover them")
        if time.monotonic() > deadline:
            raise bots.failure("did not finish in time")
        time.sleep(POLL_S)
    end = time.time()
    if not ended.exists():  # they exited before their calls ended
        _callers_exit(bots)
    return end, bots


def _callers_exit(bots: Process) -> None:
    """Wait for the callers to exit once their calls ended, and check how: with --save-audio they
    write each call's audio first, which the point's window and its last metrics leave out."""
    deadline = time.monotonic() + CALLERS_EXIT_S
    while not bots.exited():
        if time.monotonic() > deadline:
            raise bots.failure("did not exit after the calls")
        time.sleep(POLL_S)
    if bots.stop() not in (0, 3):  # 3: some turns failed, which the summary counts
        raise bots.failure("failed")


def _gateway_log(systemd: Systemd, out: Path, start: float, end: float) -> None:
    """Keep the gateway's request lines of the calls (gateway.jsonl), from which the summary takes the
    first bytes exactly; none when the gateway logs text."""
    if lines := gateway_request_lines(systemd.journal(GATEWAY, start, until=end)):
        (out / "gateway.jsonl").write_text("".join(f"{line}\n" for line in lines))


def _ended(host: Host) -> tuple[float, float]:
    """From the gateway's metrics: the requests it has ended (OUTCOMES, every class and outcome), and
    those it is relaying or holding at its admission point (BUSY)."""
    exposition = fetch(host.gateway_socket)
    ended = busy = 0.0
    for name, _, value in metric_samples(exposition):
        if name == OUTCOMES:
            ended += value
        elif name in BUSY:
            busy += value
    return ended, busy


def _check_gateway_log(
    host: Host, systemd: Systemd, base: tuple[float, tuple[float, float]]
) -> dict[str, int]:
    """Check that the gateway's journal kept a request line for every request the gateway ended since
    ``base``, the time and counters of a moment it was idle: the first bytes come from those lines, and
    journald drops what comes faster than its rate limit, which for a user unit is journald.conf's. Waits
    until the gateway is idle again, so that both counts end at the same moment."""
    since, (before, _) = base
    deadline = time.monotonic() + GATEWAY_IDLE_S
    last = None
    while True:
        ended, busy = _ended(host)
        if busy == 0 and ended == last:
            break
        if time.monotonic() > deadline:
            raise E2Error(
                f"the gateway did not go idle within {GATEWAY_IDLE_S:.0f} s after the point's tools stopped"
            )
        last = ended
        time.sleep(POLL_S)
    until = time.time()
    # Requests that went upstream with a class, as the counter counts them.
    lines = [
        record
        for at, record in logged_requests(systemd.journal(GATEWAY, since, until=until))
        if since < at <= until and "class" in record and "outcome" in record
    ]
    counted = round(ended - before)
    if len(lines) != counted:
        raise E2Error(
            f"the gateway's journal kept {len(lines)} request lines of the {counted} requests it ended "
            "during the point: journald dropped some, and the first bytes would come from the rest"
        )
    return {"request_lines": len(lines), "requests_ended": counted}


def _worker_journal(systemd: Systemd, out: Path, since: float, calls: int) -> None:
    """Keep the voice worker's journal from the start of the calls (voice-worker.log): a line per
    turn, and one per session as it ends, with its event counts. Waits until every call's session
    has ended, unless the worker logs neither kind of line."""
    deadline = time.monotonic() + SESSIONS_END_S
    while True:
        journal = systemd.journal(WORKER, since, until=time.time())
        turns, sessions = worker_lines(journal)
        if len(sessions) >= calls or not (turns or sessions) or time.monotonic() > deadline:
            break
        time.sleep(POLL_S)
    (out / "voice-worker.log").write_text(journal)


def _ready(address: str, path: str = "/ready") -> bool:
    with contextlib.suppress(httpx.HTTPError), client(address, timeout=5) as http:
        return http.get(path).status_code == 200
    return False


class _KvSampler:
    """The engine's KV use and running and waiting requests (record.KV_GAUGES) from its ``/metrics``
    every ``interval`` seconds, from its start until :meth:`stop`, a line a sample (its time, then the
    three) into ``metrics``' engine-kv.csv. A sample the engine does not answer in time is left out;
    the summary counts the samples it has. The sampler has one client for the whole run, its
    connection kept alive between samples (the engine keeps an idle one 60 s): through cyan's tunnel a
    new connection is a new SSH session on the GPU node, about 12 ms of its CPU, which a client a
    sample would cost it twice a second while it measures. A request that fails drops its client, and
    the next sample opens another."""

    def __init__(self, address: str, metrics: Path, interval: float = KV_SAMPLE_S) -> None:
        self.interval = interval
        self._stopped = threading.Event()
        metrics.mkdir(exist_ok=True)
        self._thread = threading.Thread(
            target=self._sample, args=(address, metrics / "engine-kv.csv"), daemon=True
        )
        self._thread.start()

    def _sample(self, address: str, path: Path) -> None:
        http: httpx.Client | None = None
        with path.open("w") as samples:
            samples.write("time,kv_cache_usage,running,waiting\n")
            try:
                while not self._stopped.is_set():
                    at = time.time()
                    values = None
                    try:
                        http = http or client(address, timeout=2.0)
                        response = http.get("/metrics")
                        response.raise_for_status()
                        values = kv_values(response.text)
                    except httpx.HTTPError:
                        if http is not None:
                            http.close()
                        http = None
                    if values is not None:
                        samples.write(f"{at:.6f}," + ",".join(f"{v:g}" for v in values) + "\n")
                        samples.flush()
                    self._stopped.wait(max(0.0, at + self.interval - time.time()))
            finally:
                if http is not None:
                    http.close()

    def stop(self) -> None:
        """Stop sampling once the sample in flight is written; stopping twice is stopping."""
        self._stopped.set()
        self._thread.join()


def _snapshot(host: Host, out: Path, when: str) -> float:
    """Save the gateway's ``/metrics`` (from its socket, which asks for no key) and the engine's;
    returns the time it finished."""
    metrics = out / "metrics"
    metrics.mkdir(exist_ok=True)
    files = {f"gateway-{when}.prom": host.gateway_socket, f"engine-{when}.prom": host.engine_url}
    for name, address in files.items():
        try:
            (metrics / name).write_text(fetch(address))
        except httpx.HTTPError as exc:
            raise E2Error(f"GET /metrics from {address}: {exc}") from None
    return time.time()


def _aiperf_env(arm: Arm, host: Host, out: Path, seed: int, model: str) -> dict[str, str]:
    agents = arm.agents
    data = host.root / "data" / agents.data
    if not data.is_dir():
        raise E2Error(f"no trace corpus at {data} (flatshare-bench scale-traces writes one)")
    if not (CONFIGS / "agentic_replay.yaml").is_file():
        raise E2Error(
            f"no AIPerf configuration in {CONFIGS}: install the package from its source tree"
        )
    keys = host.config / "gateway-keys.env"
    tenant_key = read_env(keys).get("FLATSHARE_GATEWAY_AGENTS_KEY") if keys.exists() else None
    if not tenant_key:
        raise E2Error(f"{keys} has no FLATSHARE_GATEWAY_AGENTS_KEY, the agents tenant's key")
    env = os.environ | read_env(CONFIGS / "aiperf.env")
    env |= {
        "FLATSHARE_BENCH_URL": f"{host.gateway_url}/v1/chat/completions",
        "FLATSHARE_BENCH_API_KEY": tenant_key,
        "FLATSHARE_BENCH_DATA": str(data),
        "FLATSHARE_BENCH_SESSIONS": str(agents.sessions),
        "FLATSHARE_BENCH_DURATION": str(AIPERF_DURATION_S),
        "FLATSHARE_BENCH_GRACE": "30",
        "FLATSHARE_BENCH_THINK_CAP": f"{agents.think_cap:g}",
        "FLATSHARE_BENCH_SEED": str(seed),
        "FLATSHARE_BENCH_CLASS": "background",
        "FLATSHARE_BENCH_ARTIFACTS": str(out / "aiperf"),
        "FLATSHARE_BENCH_MODEL": model,
        "FLATSHARE_BENCH_TOKENIZER": model,
    }
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


def _fresh(out: Path) -> None:
    run = out / "run.json"
    if run.exists():
        if orjson.loads(run.read_bytes()).get("status") == "complete":
            raise E2Error(f"{out} holds a complete point; e2 run never overwrites one")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out.rename(out.with_name(f"{out.name}.incomplete-{stamp}"))
    elif out.exists() and any(out.iterdir()):
        raise E2Error(f"{out} is not empty and holds no point")
    out.mkdir(parents=True, exist_ok=True)


class _Record:
    """run.json, rewritten at each step so that a point that dies keeps its status, and run.log."""

    def __init__(
        self, out: Path, arm: Arm, host: Host, gpu_source: Mapping[str, str | None] | None
    ) -> None:
        self.path = out / "run.json"
        self.log = out / "run.log"
        self.data: dict[str, Any] = {
            "kind": "e2-point",
            "name": arm.name,
            "status": "running",
            "host": platform.node(),
            "started": time.time(),
            "arm": asdict(arm),
            "versions": _versions(host, gpu_source),
        }
        self._write()

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._write()

    def say(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {message}"
        with self.log.open("a") as log:
            log.write(line + "\n")
        # The point goes on when the ssh session that started it closes; run.log keeps every line.
        with contextlib.suppress(OSError):
            print(line, file=sys.stderr, flush=True)

    def fail(self, exc: BaseException) -> None:
        error = str(exc) if isinstance(exc, E2Error) else f"{type(exc).__name__}: {exc}"
        self.data |= {"status": "failed", "error": error, "finished": time.time()}
        self._write()
        self.say(f"failed: {error}")

    def finish(self) -> None:
        self.data |= {"status": "complete", "finished": time.time()}
        self._write()

    def _write(self) -> None:
        _write_json(self.path, self.data)


PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"
"""flatshare-bench's own pyproject.toml, where it runs from its source tree (bootstrap.sh's editable install)."""


def _stale_install(pyproject: Path = PYPROJECT) -> str | None:
    """Why flatshare-bench's install does not match the source it runs, or None. An editable install runs the
    source tree but keeps the metadata of its install, the version run.json records among it: until
    the venv is synced again, a version change in pyproject.toml would leave every record naming the
    old one. An installed wheel has no pyproject.toml beside it and its own metadata."""
    if not pyproject.exists():
        return None
    source = tomllib.loads(pyproject.read_text())["project"]["version"]
    if (installed := metadata.version("flatshare-bench")) != source:
        return (
            f"flatshare-bench's venv was installed at version {installed} and its source is {source}: sync "
            "the venv (deploy/cyan/bootstrap.sh) so that run.json names the version that runs"
        )
    return None


def preflight_summary(path: Path) -> dict[str, Any]:
    """What a point keeps of its host's preflight: when it ran and each check that was not ok, with
    its reason and what it changes in what the results mean (docs/HARDWARE.md, the preflight)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "time": data.get("time"),
        "counts": data.get("counts"),
        "not_ok": [
            {key: check.get(key) for key in ("check", "status", "value", "reason", "effect")}
            for check in data.get("checks", [])
            if check.get("status") != "ok"
        ],
    }


def _versions(host: Host, gpu_source: Mapping[str, str | None] | None) -> dict[str, Any]:
    deployed = host.root / "src" / "DEPLOYED"  # deploy/deploy.sh records the commit it synced
    gateway = host.root / "bin" / "flatshare-gateway"
    flatshare_gateway = None
    if gateway.exists():
        shown = subprocess.run([gateway, "--version"], capture_output=True, text=True)
        flatshare_gateway = shown.stdout.strip() or None
    return {
        "source": deployed.read_text().strip() if deployed.exists() else None,
        "gpu_source": None if gpu_source is None else dict(gpu_source),
        "flatshare_bench": metadata.version("flatshare-bench"),
        "aiperf": metadata.version("aiperf"),
        "flatshare_gateway": flatshare_gateway,
    }


NODE_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
"""A GPU node's name (its host's first label, as campaign.sh gives it), which unit, file and socket
names carry."""
_REVISION = re.compile(r"[A-Za-z0-9._+-]*")
"""A deployed revision as deploy/deploy.sh writes it (``git describe --always --dirty``)."""


def gpu_sources(values: Sequence[str]) -> dict[str, str | None]:
    """``NODE=REVISION`` values (``flatshare-bench e2 run --gpu-source``) by node, an empty revision None."""
    sources: dict[str, str | None] = {}
    for value in values:
        node, sep, revision = value.partition("=")
        if not sep or not NODE_NAME.fullmatch(node) or not _REVISION.fullmatch(revision):
            raise E2Error(f"--gpu-source {value!r} is not NODE=REVISION")
        if node in sources:
            raise E2Error(f"--gpu-source names {node} twice")
        sources[node] = revision or None
    return sources


def _write_json(path: Path, data: Any) -> None:
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2) + b"\n")
