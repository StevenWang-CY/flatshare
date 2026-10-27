"""The engine a point runs on, checked before anything starts (:mod:`.run`): its own account of
itself, from its ``/server_info`` (:func:`describe_engine`; on vLLM a development route, which the
measurement drop-in serves), held against the command line that ``variant.sh use`` gives it
(:func:`check_engine`): the GPU node's engine.yaml by its hardware profile (:func:`profile_flags`),
then the variant's flags (:func:`variant_flags`) and the arm's. A point empties the engine's prefix
cache before it starts (:func:`reset_prefix_cache`).
"""

from __future__ import annotations

import json
import shlex
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from flatshare_bench.e2.arm import E2Error, Engine, flag_values
from flatshare_bench.e2.record import model_id
from flatshare_bench.e2.tools import until
from flatshare_bench.kv_metrics import fetch
from flatshare_bench.tiers import pool_tokens
from flatshare_bench.transport import client

VARIANTS = Path(__file__).resolve().parents[5] / "deploy" / "phastform" / "variants"
"""The GPU node's engine variants, from the package's source tree (``uv sync`` installs it editable):
what each adds to the engine's command line, which the engine check expects."""
PROFILES = Path(__file__).resolve().parents[5] / "deploy" / "profiles"
"""The GPU nodes' hardware profiles (deploy/profile.py renders a node's deployment from one), from the
same source tree: the model and the scheduler settings a node's engine.yaml gives its engine, which
the engine check expects."""

SGLANG = "sglang"
"""The one engine variant that serves SGLang; every other variant serves vLLM."""


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

    until(done, timeout, f"the engine did not reset its prefix cache within {timeout:.0f} s")


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
