"""flatshare-bench e2 run against fakes: the gateway and the engine on loopback, a recording systemctl,
and stand-ins for AIPerf and the callers that run as real processes, so stopping them is real."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import itertools
import json
import math
import os
import re
import signal
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from flatshare_bench.cli import main
from flatshare_bench.e2 import run as e2
from flatshare_bench.e2.arm import E2Error, Engine, Voice, flag_values, load_arm
from flatshare_bench.e2.record import current_name
from flatshare_bench.e2.run import (
    FROZEN,
    EngineInfo,
    Host,
    Process,
    Systemd,
    check_engine,
    gpu_sources,
    profile_flags,
    read_env,
    reset_cyan,
    run_point,
    toml_text,
)

FAKES = Path(__file__).parent / "fakes"
REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "deploy" / "cyan" / "gateway.example.toml"
BENCHMARKS = REPO / "benchmarks" / "e2"
RESULTS = REPO / "benchmarks" / "results" / "2026-10-23-e2"
PHASTFORM = REPO / "deploy" / "phastform"
needs_waitid = pytest.mark.skipif(not hasattr(os, "waitid"), reason=e2.NEEDS_WAITID)

ARM = """\
arm = 2
label = "stock vLLM, 4 agent sessions"

[voice]
callers = 1
turns = 3
preemptive_generation = false
events = false

[agents]
sessions = 4
warmup = 0.2
data = "corpus"

[gateway]
upstream_priority = []

[gateway.admission]
policy = "static-cap"
background_max_in_flight = 2

[engine]
flags = ["--scheduling-policy", "fcfs", "--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "0"]
"""

VOICE_ALONE = 'arm = 1\nlabel = "voice alone"\n'


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def _quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2, "POLL_S", 0.02)
    for name in ("FAKE_AIPERF", "FAKE_BOTS"):
        monkeypatch.delenv(name, raising=False)


# The arm file


def test_an_arm_file_fills_in_the_defaults(tmp_path: Path) -> None:
    arm = load_arm(write(tmp_path, "a1-voice.toml", VOICE_ALONE))
    assert (arm.name, arm.arm, arm.label) == ("a1-voice", "1", "voice alone")
    assert arm.voice == Voice(callers=2, turns=25, barge_in=0.25, preemptive_generation=True)
    assert arm.voice.warmup_turns == 0  # the arm files measured before warm-up turns stay as run
    assert (arm.agents.sessions, arm.agents.think_cap) == (0, 120.0)
    assert (arm.gateway, arm.engine, arm.router) == ({}, Engine("frozen", ()), None)


def test_an_arm_file_says_everything_about_its_point(tmp_path: Path) -> None:
    arm = load_arm(write(tmp_path, "a2-stock-s4.toml", ARM))
    assert arm.voice.preemptive_generation is False and arm.voice.barge_in == 0.25
    assert arm.voice.events is False
    assert arm.agents.warmup == 0.2 and isinstance(arm.agents.think_cap, float)
    assert arm.gateway == {
        "upstream_priority": [],
        "admission": {"policy": "static-cap", "background_max_in_flight": 2},
    }
    assert str(arm.engine) == (
        "frozen --scheduling-policy fcfs --max-num-batched-tokens 2048"
        " --long-prefill-token-threshold 0"
    )
    sub = load_arm(write(tmp_path, "a7b.toml", 'arm = "7b"\nlabel = "SGLang"\nrouter = "smg"\n'))
    assert (sub.arm, sub.router) == ("7b", "smg")


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('label = "x"\n', "arm is required"),
        ("arm = 2\n", "label is required"),
        ('arm = 2\nlabel = "x"\ncallers = 2\n', "unknown key(s): callers"),
        ('arm = 2\nlabel = "x"\n[voice]\ncaller = 2\n', "[voice]: unknown key(s): caller"),
        ('arm = 2\nlabel = "x"\n[voice]\nturns = true\n', "turns must be int, not True"),
        ('arm = 2\nlabel = "x"\n[agents]\nwarmup = "60"\n', "warmup must be float"),
        ('arm = 2\nlabel = "x"\n[voice]\npreemptive_generation = 1\n', "must be bool"),
        ('arm = 2\nlabel = "x"\n[voice]\nevents = "off"\n', "events must be bool"),
        ('arm = 11\nlabel = "x"\n', "arm is E2's arm"),
        ('arm = "7z1"\nlabel = "x"\n', "arm is E2's arm"),
        ('arm = 2\nlabel = "x"\nrouter = "nginx"\n', "router can only be smg"),
        ('arm = 2\nlabel = "x"\n[voice]\nbarge_in = 1.5\n', "barge_in is a share"),
        ('arm = 2\nlabel = "x"\n[voice]\nwarmup_turns = -1\n', "warmup_turns cannot be negative"),
        ('arm = 2\nlabel = "x"\n[engine]\nflags = ["--a b"]\n', "is not a plain word"),
        ('arm = 2\nlabel = "x"\n[engine]\nflags = ["512"]\n', "follows no flag"),
        ('arm = 2\nlabel = "x"\n[engine]\nflags = "--a"\n', "flags must be list"),
        ("arm = \n", "a2.toml"),
        (
            'arm = 2\nlabel = "x"\n[gateway.nodes.a]\nupstream = "http://127.0.0.1:1"\n',
            "[gateway.nodes] is for points on several GPU nodes, which the two-node branch keeps",
        ),
        (
            'arm = 2\nlabel = "x"\n[gateway.placement]\npolicy = "least-loaded"\n',
            "[gateway.placement] is for points on several GPU nodes",
        ),
    ],
)
def test_a_wrong_arm_file_says_what_is_wrong(tmp_path: Path, text: str, error: str) -> None:
    with pytest.raises(E2Error, match=re.escape(error)):
        load_arm(write(tmp_path, "a2.toml", text))


def campaign_lines(campaign: Path) -> list[str]:
    """A campaign file's lines as campaign.sh reads them: comments and blank lines left out."""
    lines = (line.split("#")[0].strip() for line in campaign.read_text().splitlines())
    return list(filter(None, lines))


def test_the_campaigns_name_arm_files_that_load() -> None:
    campaigns = sorted(BENCHMARKS.glob("*.campaign"))
    assert campaigns
    for campaign in campaigns:
        lines = campaign_lines(campaign)
        # The settings campaign.sh reads, each at most once: the agents' seed, the KV sampler's interval and
        # whether the calls' audio is saved.
        seeds = [line for line in lines if line.startswith("agents-seed")]
        assert len(seeds) <= 1 and all(re.fullmatch(r"agents-seed\s+\d+", s) for s in seeds)
        sampling = [line for line in lines if line.startswith("kv-sample-s")]
        assert len(sampling) <= 1 and all(
            re.fullmatch(r"kv-sample-s\s+\d+(\.\d+)?", s) for s in sampling
        )
        audio = [line for line in lines if line.startswith("save-audio")]
        assert audio in ([], ["save-audio"])
        for line in (line for line in lines if line not in seeds + sampling + audio):
            arm = load_arm(campaign.parent / line)
            variant = arm.engine.variant
            assert variant == "frozen" or (PHASTFORM / "variants" / variant).is_dir(), line
    assert all(load_arm(path) for path in (BENCHMARKS / "arms").glob("*.toml"))


V03_BUILD_SHA256 = "ba7bbe076530311b37f91aa07ac24489904a120c983f495c66f876fd7221a3cd"
V03_CONF_SHA256 = "853dacb1a096a97e2a06762ac7ca6f875c35550d3a386dc59edb2c6ab9dbee85"
"""The sha256 of make-room's build.py and flatshare-engine.conf at the v0.3 tag."""


def test_the_w5_control_is_make_room_as_v03_built_it() -> None:
    control = PHASTFORM / "variants" / "make-room-v03"
    assert hashlib.sha256((control / "build.py").read_bytes()).hexdigest() == V03_BUILD_SHA256
    # v0.3's lh-engine.conf with a first line saying what it is, its own build.py and cache directory,
    # and its module renamed, so that the engine's scheduler_cls tells it from make-room's. v0.3's
    # build.py writes the module under the name it gave it (lh_make_room_scheduler.py).
    first, rest = (control / "flatshare-engine.conf").read_text().split("\n", 1)
    assert first.startswith("# W5's control: make-room as the v0.3 tag has it")
    cache = "%h/flatshare/cache/make-room-v03"
    rename = (
        f"ExecStartPre=/bin/mv -f {cache}/lh_make_room_scheduler.py "
        f"{cache}/flatshare_make_room_v03_scheduler.py\n"
    )
    assert rest.count(rename) == 1
    # Like every engine since, it serves on its socket, in a network namespace of its own; and it
    # names the project's paths and variables as they are named since the project's rename.
    v03 = (
        rest.replace(rename, "")
        .replace(" --uds ${FLATSHARE_ENGINE_SOCKET}", "")
        .replace("%h/flatshare/src/deploy/bin/in-private-net ", "")
        .replace("flatshare_make_room_v03_scheduler", "lh_make_room_scheduler")
        .replace("make-room-v03", "make-room")
        .replace("%h/flatshare/", "%h/leaseholder/")
        .replace("$FLATSHARE_ENGINE_FLAGS", "$LH_ENGINE_FLAGS")
        .replace("Flatshare's", "Leaseholder's")
    )
    assert hashlib.sha256(v03.encode()).hexdigest() == V03_CONF_SHA256
    assert f"variants/make-room-v03/build.py {cache}" in rest
    assert f"Environment=PYTHONPATH={cache}" in rest
    assert e2.variant_flags("make-room-v03") == (
        "--scheduler-cls", "flatshare_make_room_v03_scheduler.AsyncScheduler",
        "--additional-config.make_room.max_preemptions_per_step", "2",
        "--additional-config.make_room.budget_cap", "true",
        "--additional-config.make_room.reserve_tokens", "2048",
    )  # fmt: skip


def test_the_engine_check_tells_the_w5_control_from_make_room() -> None:
    # The control runs make-room's switches under a scheduler class of its own, so that a point on the
    # other module fails before anything starts.
    switches = {"make_room": MAKE_ROOM}
    final = vllm(switches, scheduler_cls=MAKE_ROOM_CLS)
    control = vllm(switches, scheduler_cls="flatshare_make_room_v03_scheduler.AsyncScheduler")
    check_engine(Engine("make-room", ()), final)
    check_engine(Engine("make-room-v03", ()), control)
    with pytest.raises(E2Error, match=r"use make-room-v03$"):
        check_engine(Engine("make-room-v03", ()), final)
    with pytest.raises(E2Error, match=r"use make-room$"):
        check_engine(Engine("make-room", ()), control)


W5 = ("w5-a1-voice", "w5-disc-frozen-s24", "w5-disc-pr-ab48-s24", "w5-disc-pr-ab48-v03-s24")
"""W5's four configurations, each the W4 one it re-measures with a warm-up turn opening each call."""
W4 = {
    "w5-a1-voice": "a1-voice",
    "w5-disc-frozen-s24": "disc-frozen-s24",
    "w5-disc-pr-ab48-s24": "disc-pr-ab48-s24",
    "w5-disc-pr-ab48-v03-s24": "disc-pr-ab48-s24",
}


def test_the_w5_campaign_rotates_its_four_configurations_over_five_seeded_rounds() -> None:
    rounds = [campaign_lines(BENCHMARKS / f"w5-r{r}.campaign") for r in range(1, 6)]
    assert [lines[0] for lines in rounds] == [f"agents-seed {seed}" for seed in range(11, 16)]
    orders = [[Path(line).stem for line in lines[1:]] for lines in rounds]
    assert all(sorted(order) == sorted(W5) for order in orders)
    for order, then in itertools.pairwise(orders):
        assert then == [*order[1:], order[0]]  # rotated by one from round to round
    assert all({order[place] for order in orders} == set(W5) for place in range(4))
    for name in W5:
        arm, before = (
            load_arm(BENCHMARKS / "arms" / f"{name}.toml"),
            load_arm(BENCHMARKS / "arms" / f"{W4[name]}.toml"),
        )
        assert arm.voice == replace(before.voice, warmup_turns=1)
        assert (arm.agents, arm.gateway, arm.router) == (
            before.agents,
            before.gateway,
            before.router,
        )
        assert arm.engine.flags == before.engine.flags
    variants = {name: load_arm(BENCHMARKS / "arms" / f"{name}.toml").engine.variant for name in W5}
    assert variants["w5-disc-pr-ab48-s24"] == "make-room"
    assert variants["w5-disc-pr-ab48-v03-s24"] == "make-room-v03"


W5_PR = {"w5-disc-pr-a-s24": "disc-pr-a-s24", "w5-disc-pr-ab-s24": "disc-pr-ab-s24"}
"""The PRs' other two configurations W5 re-measures, each with the W4 one it re-measures."""
W5_PR_ROUNDS = ("w5-disc-frozen-s24", *W5_PR, "w5-disc-pr-ab48-s24")
"""The PR campaign's four configurations: the two, and two of W5's campaign beside them."""


def test_the_w5_pr_campaign_rotates_the_prs_configurations_beside_the_frozen_box() -> None:
    rounds = [campaign_lines(BENCHMARKS / f"w5-pr-r{r}.campaign") for r in range(1, 6)]
    assert [lines[0] for lines in rounds] == [f"agents-seed {seed}" for seed in range(11, 16)]
    orders = [[Path(line).stem for line in lines[1:]] for lines in rounds]
    assert orders[0] == list(W5_PR_ROUNDS)
    for order, then in itertools.pairwise(orders):
        assert then == [*order[1:], order[0]]  # rotated by one from round to round
    assert all({order[place] for order in orders} == set(W5_PR_ROUNDS) for place in range(4))
    for name, w4 in W5_PR.items():
        arm, before = (
            load_arm(BENCHMARKS / "arms" / f"{name}.toml"),
            load_arm(BENCHMARKS / "arms" / f"{w4}.toml"),
        )
        assert arm.voice == replace(before.voice, warmup_turns=1)
        # The same agents, gateway, router and engine: make-room, now as W5's fixes leave it.
        assert (arm.agents, arm.gateway, arm.router, arm.engine) == (
            before.agents,
            before.gateway,
            before.router,
            before.engine,
        )


W5_EXT = {
    "w5-disc-a8c-resubmit-s24": "disc-a8c-resubmit-s24",
    "w5-std-frozen-s16": "a4-b512-t256-s16",
    "w5-std-pr-ab48-s16": "a4-b512-t256-s16",
}
"""The W5 extension's new configurations, each with the W4 arm whose agents, gateway and router it keeps."""
W5_EXT_GROUPS = (
    ("w5-disc-a8c-resubmit-s24",),
    ("w5-std-frozen-s16",),
    ("w5-std-pr-ab48-s16", "w5-disc-pr-ab48-s24"),
)
"""Its engine groups, in round 1's order: the PRs' two points share their engine, the second W5's own."""


def test_the_w5_extension_rotates_its_engine_groups_over_w5s_seeds() -> None:
    rounds = [campaign_lines(BENCHMARKS / f"w5-ext-r{r}.campaign") for r in range(1, 6)]
    assert [lines[0] for lines in rounds] == [f"agents-seed {seed}" for seed in range(11, 16)]
    orders = [[Path(line).stem for line in lines[1:]] for lines in rounds]
    for r, order in enumerate(orders):
        groups = W5_EXT_GROUPS[r % 3 :] + W5_EXT_GROUPS[: r % 3]  # rotated by one group a round
        assert order == [point for group in groups for point in group]
    for name, w4 in W5_EXT.items():
        arm, before = (
            load_arm(BENCHMARKS / "arms" / f"{name}.toml"),
            load_arm(BENCHMARKS / "arms" / f"{w4}.toml"),
        )
        assert arm.voice == replace(before.voice, warmup_turns=1)
        assert (arm.agents, arm.gateway, arm.router) == (
            before.agents,
            before.gateway,
            before.router,
        )
    # The standard scenario's PR point runs the engine of W5's discriminating one, so the two need no switch.
    engine = load_arm(BENCHMARKS / "arms" / "w5-std-pr-ab48-s16.toml").engine
    assert engine == load_arm(BENCHMARKS / "arms" / "w5-disc-pr-ab48-s24.toml").engine
    assert (
        load_arm(BENCHMARKS / "arms" / "w5-disc-a8c-resubmit-s24.toml").engine.variant == "frozen"
    )


W5_CAP = {
    "w5-a1-voice-c4": ("w5-a1-voice", 4),
    "w5-a1-voice-c6": ("w5-a1-voice", 6),
    "w5-disc-pr-ab48-c4-s24": ("w5-disc-pr-ab48-s24", 4),
    "w5-disc-pr-ab48-c6-s24": ("w5-disc-pr-ab48-s24", 6),
}
"""The calls-per-GPU configurations: each a W5 arm with more callers."""


def test_the_calls_per_gpu_rounds_alternate_their_engine_groups_and_caller_counts() -> None:
    rounds = [campaign_lines(BENCHMARKS / f"w5-cap-r{r}.campaign") for r in range(1, 4)]
    assert [lines[0] for lines in rounds] == [f"agents-seed {seed}" for seed in range(11, 14)]
    orders = [[Path(line).stem for line in lines[1:]] for lines in rounds]
    assert all(sorted(order) == sorted(W5_CAP) for order in orders)
    for order, then in itertools.pairwise(orders):
        assert then == order[::-1]  # the groups and the caller counts in each swap places
    for name, (w5, callers) in W5_CAP.items():
        arm, before = (
            load_arm(BENCHMARKS / "arms" / f"{name}.toml"),
            load_arm(BENCHMARKS / "arms" / f"{w5}.toml"),
        )
        assert arm.voice == replace(before.voice, callers=callers)
        assert (arm.agents, arm.gateway, arm.router, arm.engine) == (
            before.agents,
            before.gateway,
            before.router,
            before.engine,
        )


RS = ("w5-disc-frozen-s24", "w5-disc-pr-ab48-s24")
"""RS's two configurations: W5's arm files of the frozen box and of the PRs' 48-token configuration."""


def test_rs_samples_the_kv_beside_w5s_two_configurations_in_alternating_order() -> None:
    # RS measures the box's running share as the A100's core measured its own: W5's arm files, round N on
    # the agents' sample of W5's round N, the engine's free KV sampled every 0.5 s.
    rounds = [campaign_lines(BENCHMARKS / f"rs-r{r}.campaign") for r in range(1, 6)]
    assert [lines[:2] for lines in rounds] == [
        [f"agents-seed {seed}", "kv-sample-s 0.5"] for seed in range(11, 16)
    ]
    orders = [[Path(line).stem for line in lines[2:]] for lines in rounds]
    assert orders == [list(RS) if r % 2 else list(RS[::-1]) for r in range(1, 6)]


def test_flag_values_pair_each_flag_with_its_value() -> None:
    flags = ["--max-num-seqs", "12", "--no-async-scheduling", "--watermark", "0.1", "--x"]
    assert flag_values(flags) == {
        "--max-num-seqs": "12",
        "--no-async-scheduling": "true",
        "--watermark": "0.1",
        "--x": "true",
    }


# The gateway's configuration and the environment files


def test_the_gateway_configuration_merges_the_arm_and_round_trips(tmp_path: Path) -> None:
    base = tomllib.loads(EXAMPLE.read_text())
    overlay = {
        "upstream": "unix:/run/user/1007/flatshare/smg.sock",
        "upstream_priority": ["smg-header"],
        "admission": {"policy": "flow-control", "ttft_target_ms": {"realtime": 300.0}},
        "tenants": {"agents": {"max_class": "background"}},
    }
    merged = e2._merge(base, overlay)
    assert tomllib.loads(toml_text(merged)) == merged
    assert merged["tenants"]["agents"]["trusted"] is True  # tables merge key by key
    assert merged["tenants"]["agents"]["max_class"] == "background"
    assert merged["listen"] == base["listen"]  # [[listen]] stays an array of tables
    tricky = {
        "text": 'quote " slash \\ tab \t del \x7f ünï',
        "odd key": 1,
        "nested": {"a.b": [1.5, 2]},
    }
    assert tomllib.loads(toml_text(tricky)) == tricky
    redacted = e2._redacted(merged)
    assert redacted["tenants"]["agents"]["keys_sha256"] == ["<redacted>"]
    assert "<sha256" not in json.dumps(redacted)


def test_env_files_read_like_the_shell_reads_them(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "a.env",
        "# comment\nexport A=1\nB=\"two words\"\nC='x'\n\nD=a=b\n# export E=5\nnot an assignment\n",
    )
    assert read_env(path) == {"A": "1", "B": "two words", "C": "x", "D": "a=b"}


# The engine


FROZEN_SCHEDULER = {
    "policy": "priority",
    "max_num_batched_tokens": 512,
    "long_prefill_token_threshold": 256,
    "scheduler_cls": None,  # vLLM's own
}
STOCK_SCHEDULER = {
    "policy": "fcfs",
    "max_num_batched_tokens": 2048,
    "long_prefill_token_threshold": 0,
}


def vllm(additional_config: dict[str, Any] | None = None, **scheduler: Any) -> EngineInfo:
    """A vLLM engine as its /server_info reports it: the frozen box, with ``scheduler`` settings and
    ``additional_config`` changed."""
    settings = FROZEN_SCHEDULER | {"watermark": 0.0} | scheduler
    return EngineInfo(
        "vllm", "0.32.0", 36400, settings, {"additional_config": additional_config or {}}
    )


def test_the_engine_check_compares_each_flag_with_the_engines_setting(tmp_path: Path) -> None:
    engine = Engine("frozen", ("--max-num-batched-tokens", "1024", "--watermark", "0", "--kv-x"))
    frozen = ["--scheduling-policy", "--max-num-batched-tokens", "--long-prefill-token-threshold"]
    assert check_engine(engine, vllm(max_num_batched_tokens=1024)) == {
        "checked": [*frozen, "--watermark"],
        "unchecked": ["--kv-x"],
    }
    with pytest.raises(E2Error, match="max_num_batched_tokens = 512, the point needs --max-num"):
        check_engine(engine, vllm())
    with pytest.raises(E2Error, match="the engine is vllm, the arm file needs sglang"):
        check_engine(Engine("sglang", ()), vllm())
    variants = write_variant(tmp_path, "sglang", "python -m sglang.launch_server --page-size 16")
    sglang = EngineInfo("sglang", "0.5.22", 36400, {"chunked_prefill_size": 512}, {})
    flags = ("--chunked-prefill-size", "512")
    assert check_engine(Engine("sglang", flags), sglang, variants) == {
        "checked": ["--chunked-prefill-size"],
        "unchecked": ["--page-size"],  # the variant's own flag, which this engine does not report
    }


def write_variant(variants: Path, name: str, command: str) -> Path:
    """An engine variant whose flatshare-engine.conf starts ``command`` as the frozen box's does, before the
    arm's flags; returns the variants directory."""
    (variants / name).mkdir(parents=True)
    (variants / name / "flatshare-engine.conf").write_text(
        f"# Variant {name}\n[Service]\nExecStart=\nExecStart={command} $FLATSHARE_ENGINE_FLAGS\n"
    )
    return variants


VLLM_SERVE = (
    "%h/flatshare/venvs/engine/bin/vllm serve --config %h/flatshare/src/engine.yaml "
    "--uds ${FLATSHARE_ENGINE_SOCKET}"
)
MAKE_ROOM = {"max_preemptions_per_step": 2, "budget_cap": True, "reserve_tokens": 2048}
MAKE_ROOM_CLS = "flatshare_make_room_scheduler.AsyncScheduler"


def variants_of_the_box(tmp_path: Path) -> Path:
    """make-room and pr54530 as the frozen box's variants start the engine, and mps, which leaves its
    command line alone."""
    write_variant(
        tmp_path,
        "make-room",
        f"{VLLM_SERVE} --scheduler-cls {MAKE_ROOM_CLS} "
        "--additional-config.make_room.max_preemptions_per_step 2 \\\n"
        "  --additional-config.make_room.budget_cap true "
        "--additional-config.make_room.reserve_tokens 2048",
    )
    write_variant(
        tmp_path,
        "pr54530",
        f"{VLLM_SERVE} --scheduler-cls flatshare_pr54530_scheduler.AsyncScheduler --max-num-seqs 12",
    )
    (tmp_path / "mps").mkdir()
    (tmp_path / "mps" / "flatshare-stt.conf").write_text("[Service]\nEnvironment=X=1\n")
    return tmp_path


def test_a_variants_flags_are_those_of_the_engine_command_it_starts(tmp_path: Path) -> None:
    variants = variants_of_the_box(tmp_path)
    assert e2.variant_flags("make-room", variants) == (
        "--scheduler-cls", MAKE_ROOM_CLS,
        "--additional-config.make_room.max_preemptions_per_step", "2",
        "--additional-config.make_room.budget_cap", "true",
        "--additional-config.make_room.reserve_tokens", "2048",
    )  # fmt: skip
    assert e2.variant_flags("frozen", variants) == e2.variant_flags("mps", variants) == ()
    with pytest.raises(E2Error, match="no engine variant nope in"):
        e2.variant_flags("nope", variants)
    # The repository's variants: each names its scheduler class, and its flags pair up.
    for variant in (PHASTFORM / "variants").iterdir():
        if variant.is_dir():
            flag_values(e2.variant_flags(variant.name))
    assert dict(pairs(e2.variant_flags("make-room")))["--scheduler-cls"] == MAKE_ROOM_CLS
    assert dict(pairs(e2.variant_flags("pr54530"))) == {
        "--scheduler-cls": "flatshare_pr54530_scheduler.AsyncScheduler",
        "--max-num-seqs": "12",
    }


def pairs(flags: tuple[str, ...]) -> list[tuple[str, str]]:
    return list(flag_values(flags).items())


def test_the_engine_check_tells_the_variants_apart(tmp_path: Path) -> None:
    variants = variants_of_the_box(tmp_path)
    frozen = vllm()
    make_room = vllm({"make_room": MAKE_ROOM}, scheduler_cls=MAKE_ROOM_CLS)
    pr54530 = vllm(scheduler_cls="flatshare_pr54530_scheduler.AsyncScheduler", max_num_seqs=12)
    engines = {"frozen": frozen, "make-room": make_room, "pr54530": pr54530}
    for variant, info in engines.items():
        check_engine(Engine(variant, ()), info, variants)  # each passes for itself
    assert check_engine(Engine("make-room", ()), make_room, variants)["checked"] == [
        "--scheduling-policy",
        "--max-num-batched-tokens",
        "--long-prefill-token-threshold",
        "--scheduler-cls",
        "--additional-config.make_room.max_preemptions_per_step",
        "--additional-config.make_room.budget_cap",
        "--additional-config.make_room.reserve_tokens",
    ]
    for variant in engines:
        for other, info in engines.items():
            if other != variant:  # and none for another
                with pytest.raises(E2Error, match=f"use {variant}$"):
                    check_engine(Engine(variant, ()), info, variants)
    with pytest.raises(
        E2Error, match=f"scheduler_cls = None, the point needs --scheduler-cls {MAKE_ROOM_CLS}"
    ):
        check_engine(Engine("make-room", ()), frozen, variants)
    with pytest.raises(
        E2Error, match=f"scheduler_cls = '{MAKE_ROOM_CLS}', the point needs vLLM's own"
    ):
        check_engine(Engine("frozen", ()), make_room, variants)


def test_the_engine_runs_exactly_the_switches_its_flags_set(tmp_path: Path) -> None:
    variants = variants_of_the_box(tmp_path)
    no_reserve = Engine("make-room", ("--additional-config.make_room.reserve_tokens", "0"))
    reserve0 = vllm({"make_room": MAKE_ROOM | {"reserve_tokens": 0}}, scheduler_cls=MAKE_ROOM_CLS)
    assert (
        "--additional-config.make_room.reserve_tokens"
        in (check_engine(no_reserve, reserve0, variants)["checked"])
    )
    reserve = vllm({"make_room": MAKE_ROOM}, scheduler_cls=MAKE_ROOM_CLS)
    window = vllm({"make_room": MAKE_ROOM | {"budget_cap_tokens": 48}}, scheduler_cls=MAKE_ROOM_CLS)
    unset = vllm({"make_room": {"budget_cap": True}}, scheduler_cls=MAKE_ROOM_CLS)
    for engine, info, error in (
        # an ablation on an engine that still runs the switch it turns off
        (no_reserve, reserve, "runs additional_config make_room.reserve_tokens = 2048, the point "
         "needs --additional-config.make_room.reserve_tokens 0"),
        # arm 9 on an engine left with a 48-token window
        (Engine("make-room", ()), window, "runs additional_config make_room.budget_cap_tokens = 48, "
         "which the point's flags do not set"),
        (Engine("make-room", ()), unset, "additional_config has no "
         "make_room.max_preemptions_per_step, the point needs "
         "--additional-config.make_room.max_preemptions_per_step 2"),
        # the frozen box's scheduler, with make-room's switches left over: they would act on nothing,
        # but the engine is not the one the arm file names
        (Engine("frozen", ()), vllm({"make_room": MAKE_ROOM}), "runs additional_config make_room = "
         "{'max_preemptions_per_step': 2, 'budget_cap': True, 'reserve_tokens': 2048}, which the "
         "point's flags do not set"),
        # flags that set a table and a value in the same place
        (Engine("make-room", ("--additional-config.make_room", "0")), reserve,
         "--additional-config.make_room sets a value where other flags set keys"),
        (Engine("make-room", ("--additional-config.make_room.budget_cap.x", "0")), reserve,
         "--additional-config.make_room.budget_cap.x sets a key inside a value of another flag"),
        # a value where the point sets a table
        (Engine("make-room", ()), vllm({"make_room": 0}, scheduler_cls=MAKE_ROOM_CLS), "runs "
         "additional_config make_room = 0, the point needs "
         "--additional-config.make_room.max_preemptions_per_step 2 "
         "--additional-config.make_room.budget_cap true "
         "--additional-config.make_room.reserve_tokens 2048"),
    ):  # fmt: skip
        with pytest.raises(E2Error, match=re.escape(error)):
            check_engine(engine, info, variants)
    # An engine that reports neither leaves them unchecked.
    bare = EngineInfo(
        "vllm",
        "0.32.0",
        36400,
        {k: v for k, v in reserve.settings.items() if k != "scheduler_cls"},
        {},
    )
    assert check_engine(no_reserve, bare, variants)["unchecked"] == [
        "--scheduler-cls",
        "--additional-config.make_room.max_preemptions_per_step",
        "--additional-config.make_room.budget_cap",
        "--additional-config.make_room.reserve_tokens",
    ]


def test_the_committed_points_ran_the_engines_their_arm_files_name() -> None:
    """Each vLLM point of the committed records passes the engine check of its arm file, its scheduler
    class among what is checked: the check refuses none of the engines the variants start."""
    checked = 0
    for engine_json in sorted(RESULTS.glob("*/*/engine.json")):
        config = json.loads(engine_json.read_text())
        if "scheduler_config" not in config:
            continue  # SGLang's
        arm = load_arm(engine_json.with_name("arm.toml"))
        settings = config["cache_config"] | config["scheduler_config"]
        # The records name the scheduler's module as it was named before the project's rename.
        if settings.get("scheduler_cls"):
            settings["scheduler_cls"] = current_name(settings["scheduler_cls"])
        flags = check_engine(arm.engine, EngineInfo("vllm", "0.32.0", 36400, settings, config))
        assert "--scheduler-cls" not in flags["unchecked"], engine_json
        checked += 1
    assert checked == 119


def test_a_vllm_point_needs_the_frozen_settings_its_flags_keep(tmp_path: Path) -> None:
    with pytest.raises(E2Error, match="policy = 'fcfs', the point needs --scheduling-policy prio"):
        check_engine(Engine("frozen", ()), vllm(**STOCK_SCHEDULER))  # left over from arm 2
    variants = variants_of_the_box(tmp_path)
    pr54530 = vllm(scheduler_cls="flatshare_pr54530_scheduler.AsyncScheduler", max_num_seqs=12)
    assert check_engine(Engine("pr54530", ()), pr54530, variants)["unchecked"] == []
    engine_yaml = (PHASTFORM / "engine.yaml").read_text()
    for flag, value in flag_values(e2.FROZEN).items():
        assert re.search(rf"^{flag.removeprefix('--')}: {value}$", engine_yaml, re.M), flag


def rendered_engine_yaml(tmp_path: Path, name: str) -> str:
    """A profile's engine.yaml as deploy/profile.py renders it, on an A100 host's facts."""
    facts = json.loads((FAKES / "a100-host.json").read_text())
    host_file = tmp_path / "host.json"
    host_file.write_text(json.dumps(facts))
    out = tmp_path / name
    render = [sys.executable, str(REPO / "deploy" / "profile.py"), "render", name]
    done = subprocess.run(
        [*render, "--host", str(host_file), "--out", str(out)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return (out / "engine.yaml").read_text()


def test_a_profile_gives_the_engine_settings_its_engine_yaml_sets(tmp_path: Path) -> None:
    box = profile_flags("rtx-5060-ti")
    assert box[: len(FROZEN)] == FROZEN  # the frozen box's scheduler settings, then its model
    assert flag_values(box[len(FROZEN) :]) == {
        "--model": "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "--revision": REVISION,
        "--max-model-len": "32768",
    }
    for name in ("rtx-5060-ti", "a100-40gb"):
        engine_yaml = (
            (PHASTFORM / "engine.yaml").read_text()
            if name == "rtx-5060-ti"
            else rendered_engine_yaml(tmp_path, name)
        )
        for flag, value in flag_values(profile_flags(name)).items():
            line = rf"^{flag.removeprefix('--')}: {re.escape(value)}$"
            assert re.search(line, engine_yaml, re.M), (name, flag)
    with pytest.raises(E2Error, match="no hardware profile h100"):
        profile_flags("h100")


def test_the_engine_check_holds_the_profiles_model_and_the_kernels(tmp_path: Path) -> None:
    box = profile_flags("rtx-5060-ti")
    model = {
        "model": "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "revision": REVISION,
        "max_model_len": 32768,
    }
    engine = vllm(**model, linear_backend="auto")
    frozen = ["--scheduling-policy", "--max-num-batched-tokens", "--long-prefill-token-threshold"]
    assert check_engine(Engine("frozen", ()), engine, base=box) == {
        "checked": [*frozen, "--model", "--revision", "--max-model-len"],
        "unchecked": [],
    }
    # Another model, as an arm that switches it names it, against the box's.
    other = ("--model", "Qwen/Qwen3-14B-FP8", "--revision", "abc", "--max-model-len", "32768")
    with pytest.raises(
        E2Error,
        match="model = 'Qwen/Qwen3-4B-Instruct-2507-FP8', the point needs --model Qwen/Qwen3-14B",
    ):
        check_engine(Engine("frozen", other), engine, base=box)
    # The linear kernel an arm asks for is kernel_config's.
    marlin = Engine("frozen", ("--linear-backend", "marlin"))
    assert (
        "--linear-backend"
        in check_engine(marlin, vllm(**model, linear_backend="marlin"), base=box)["checked"]
    )
    with pytest.raises(
        E2Error, match="linear_backend = 'auto', the point needs --linear-backend marlin"
    ):
        check_engine(marlin, engine, base=box)


def test_an_engines_settings_name_its_model_by_its_id_and_hold_its_kernels() -> None:
    engine = fake_engine(dict(FROZEN_SCHEDULER))
    try:
        info = e2.describe_engine(engine.url)
    finally:
        engine.shutdown()
    # Offline mode reports the model by its path in the cache; the settings name it by its id.
    assert (info.settings["model"], info.settings["revision"]) == (
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        REVISION,
    )
    assert info.settings["max_model_len"] == 32768 and info.settings["linear_backend"] == "auto"
    assert info.settings["policy"] == "priority" and info.pool_tokens == 36400


def test_a_host_file_sets_the_hosts_fields(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert Host.load(home) == Host(home=home)  # cyan: no file, the defaults
    path = home / ".config" / "flatshare" / "e2-host.toml"
    path.parent.mkdir(parents=True)
    path.write_text('profile = "a100-40gb"\naiperf_cores = "21-27"\ncaller_cores = "28-31"\n')
    host = Host.load(home)
    assert (host.profile, host.aiperf_cores, host.caller_cores) == ("a100-40gb", "21-27", "28-31")
    assert host.engine_url == Host().engine_url and host.home == home
    path.write_text('profile = "a100-40gb"\nengine_cores = "0-7"\n')
    with pytest.raises(E2Error, match="sets engine_cores; a host sets"):
        Host.load(home)
    path.write_text("aiperf_cores = 21\n")
    with pytest.raises(E2Error, match="aiperf_cores must be strings"):
        Host.load(home)


# Processes


@needs_waitid
def test_stopping_a_tool_kills_its_whole_process_group(tmp_path: Path) -> None:
    artifacts = tmp_path / "aiperf"
    env = os.environ | {"FLATSHARE_BENCH_ARTIFACTS": str(artifacts), "FAKE_AIPERF": "stubborn"}
    command = [sys.executable, str(FAKES / "fake_aiperf.py")]
    tool = Process("AIPerf", command, tmp_path / "log", env, str(tmp_path))
    assert tool.mark.startswith(f"{tmp_path}/")  # its scope, the test's own
    child = int(wait_for(artifacts / "child.pid").read_text())
    assert not tool.exited()
    assert tool.stop(grace=0.3) == -9  # it ignored SIGINT, so it was killed
    assert gone(child)  # and so was the process it started
    assert tool.exited() and tool.stop() == -9


PROC = pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="reads /proc (Linux)")


@PROC
@needs_waitid
def test_stopping_a_tool_kills_what_it_started_in_a_session_of_its_own(tmp_path: Path) -> None:
    """AIPerf starts some services with setsid, out of its process group: its mark reaches them."""
    artifacts = tmp_path / "aiperf"
    env = os.environ | {"FLATSHARE_BENCH_ARTIFACTS": str(artifacts), "FAKE_AIPERF": "orphan"}
    command = [sys.executable, str(FAKES / "fake_aiperf.py")]
    tool = Process("AIPerf", command, tmp_path / "log", env, str(tmp_path))
    child = int(wait_for(artifacts / "child.pid").read_text())
    assert os.getpgid(child) != tool.popen.pid
    assert tool.stop(grace=5) == 0  # it stopped on SIGINT
    assert gone(child)


@PROC
@needs_waitid
def test_a_run_ends_the_process_group_a_killed_run_left_although_its_titles_blanked_their_marks(
    tmp_path: Path,
) -> None:
    """AIPerf's services set their titles (setproctitle), which blanks /proc/PID/environ and the mark in it,
    but stay in AIPerf's process group, which is named in the run's scope until the tool is stopped: a
    run killed before it could stop its tools leaves both running, and the next run's start ends them."""
    pytest.importorskip("setproctitle")
    artifacts = tmp_path / "aiperf"
    env = os.environ | {"FLATSHARE_BENCH_ARTIFACTS": str(artifacts), "FAKE_AIPERF": "titled"}
    command = [sys.executable, str(FAKES / "fake_aiperf.py")]
    tool = Process("AIPerf", command, tmp_path / "log", env, str(tmp_path))
    child = int(wait_for(artifacts / "child.pid").read_text())
    try:
        assert os.getpgid(child) == tool.popen.pid
        titled(child)
        assert (tmp_path / e2.TOOLS / str(tool.popen.pid)).exists()
        # The run is killed here, before stop(); the next run's start:
        assert sorted(e2.kill_left(str(tmp_path))) == sorted([tool.popen.pid, child])
        assert gone(child) and tool.popen.wait(timeout=5) == -9
        assert list((tmp_path / e2.TOOLS).iterdir()) == []
    finally:
        for pid in (child, tool.popen.pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@needs_waitid
def test_a_stopped_tool_leaves_no_group_named_in_its_scope(tmp_path: Path) -> None:
    artifacts = tmp_path / "aiperf"
    env = os.environ | {"FLATSHARE_BENCH_ARTIFACTS": str(artifacts)}
    tool = Process(
        "AIPerf",
        [sys.executable, str(FAKES / "fake_aiperf.py")],
        tmp_path / "log",
        env,
        str(tmp_path),
    )
    wait_for(artifacts / "profile_export.jsonl")
    assert tool.stop(grace=5) == 0
    assert list((tmp_path / e2.TOOLS).iterdir()) == [] and e2.kill_left(str(tmp_path)) == []


def titled(pid: int) -> None:
    """Wait for process ``pid`` to set its title, which blanks its environment and the mark in it."""
    deadline = time.monotonic() + 10
    while e2.TOOL_MARK.encode() in Path(f"/proc/{pid}/environ").read_bytes():
        assert time.monotonic() < deadline, f"{pid} never set its title"
        time.sleep(0.05)


def leaderless(scope: Path) -> tuple[Path, int]:
    """A tool in ``scope`` as a run killed before it could stop it may leave it: its leader exited and
    reaped, and its titled child running on in its group, which keeps the group's number taken. Returns
    the tool's record and the child's pid."""
    pytest.importorskip("setproctitle")
    artifacts = scope / "aiperf"
    env = os.environ | {"FLATSHARE_BENCH_ARTIFACTS": str(artifacts), "FAKE_AIPERF": "titled"}
    command = [sys.executable, str(FAKES / "fake_aiperf.py")]
    tool = Process("AIPerf", command, scope / "log", env, str(scope))
    child = int(wait_for(artifacts / "child.pid").read_text())
    titled(child)
    os.kill(tool.popen.pid, signal.SIGINT)  # which the child ignores
    assert tool.popen.wait(timeout=10) == 0
    return scope / e2.TOOLS / str(tool.popen.pid), child


@PROC
@needs_waitid
def test_a_run_ends_the_titled_processes_of_a_tools_group_whose_leader_is_gone(
    tmp_path: Path,
) -> None:
    """The members of a group whose leader has exited keep its number taken: those titled in the
    cgroup the tool started in are the tool's."""
    named, child = leaderless(tmp_path)
    try:
        assert e2.kill_left(str(tmp_path)) == [child]
        assert gone(child) and not named.exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


@PROC
@needs_waitid
@pytest.mark.parametrize("which", ["empty", "of an earlier boot", "of another cgroup"])
def test_a_record_that_does_not_name_the_tool_leaves_the_titled_processes_of_its_group(
    tmp_path: Path, which: str
) -> None:
    """A record cut short names nothing, and a pid of an earlier boot nothing of this one. A titled
    process in another cgroup is another's: vLLM's engine core reads blank too, and on a single host it
    runs as flatshare-bench's user, in its unit's cgroup under systemd."""
    named, child = leaderless(tmp_path)
    started = named.read_text().partition("\n")[0]
    cgroup = Path(f"/proc/{child}/cgroup").read_text()
    unit = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/flatshare-engine.service\n"
    records = {
        "empty": "",
        "of an earlier boot": f"an-earlier-boot/{started.split('/')[1]}\n{cgroup}",
        "of another cgroup": f"{started}\n{unit}",
    }
    named.write_text(records[which])
    try:
        assert e2.kill_left(str(tmp_path)) == [] and not gone(child, timeout=0.2)
        assert not named.exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


def name_group(scope: Path, pgid: int, started: str | None) -> Path:
    """Name process group ``pgid`` in ``scope`` as Process names a tool's: ``started`` for its leader's
    start, and the cgroup of this process, which the processes it starts share."""
    named = scope / e2.TOOLS / str(pgid)
    named.parent.mkdir(exist_ok=True)
    named.write_text(f"{started}\n{Path('/proc/self/cgroup').read_text()}")
    return named


@PROC
def test_a_record_leaves_a_process_that_took_its_number_since(tmp_path: Path) -> None:
    """A killed run's record outlives its tool's group, whose number may be another's by the next run's
    start: here the leader of a group of its own, started after the tool."""
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        named = name_group(tmp_path, other.pid, e2._started(os.getpid()))  # the tool's, earlier
        assert e2.kill_left(str(tmp_path)) == [] and other.poll() is None
        assert not named.exists()
    finally:
        other.kill()


@PROC
def test_a_record_leaves_a_group_that_took_its_number_since_and_lost_its_own_leader(
    tmp_path: Path,
) -> None:
    """The group that took a record's number may have lost its own leader too, as a shell's pipeline
    can, or a daemon's that forked: its members' environments read without the mark."""
    shell = subprocess.Popen(
        ["sh", "-c", "sleep 60 >/dev/null & echo $!"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    member = int(shell.communicate()[0])  # the shell has exited, and been reaped
    try:
        assert os.getpgid(member) == shell.pid
        named = name_group(tmp_path, shell.pid, e2._started(os.getpid()))
        assert e2.kill_left(str(tmp_path)) == [] and not gone(member, timeout=0.2)
        assert not named.exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(member, signal.SIGKILL)


@PROC
def test_a_record_reads_a_process_whose_name_the_kernel_cut_inside_a_character(
    tmp_path: Path,
) -> None:
    """The kernel keeps 15 bytes of a process's name, which can end inside a character, and then
    /proc/PID/stat is not UTF-8. A file in the directory that no pid names is no record."""
    name = ("abcdefghijklmn" + chr(0xE9)).encode()  # 16 bytes: é is two
    program = (
        "import ctypes, sys, time\n"
        "ctypes.CDLL(None).prctl(15, bytes.fromhex(sys.argv[1]), 0, 0, 0)  # PR_SET_NAME\n"
        "time.sleep(60)"
    )
    cut = subprocess.Popen([sys.executable, "-c", program, name.hex()], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while Path(f"/proc/{cut.pid}/comm").read_bytes() != name[:15] + b"\n":
            assert time.monotonic() < deadline, "the process never took its name"
            time.sleep(0.02)
        name_group(tmp_path, cut.pid, e2._started(os.getpid()))  # another start
        assert e2.kill_left(str(tmp_path)) == [] and cut.poll() is None
        name_group(tmp_path, cut.pid, e2._started(cut.pid))
        write(tmp_path / e2.TOOLS, "1.part", "")
        assert e2.kill_left(str(tmp_path)) == [cut.pid] and cut.wait(timeout=5) == -9
        assert [path.name for path in (tmp_path / e2.TOOLS).iterdir()] == ["1.part"]
    finally:
        cut.kill()


@PROC
@needs_waitid
def test_a_tool_that_cannot_be_named_is_not_left_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing would stop a tool that no Process names: a scope that cannot hold its record starts
    none, and a record that cannot be written (a full disk, say) ends it before the error goes on."""
    env = os.environ | {"FLATSHARE_BENCH_ARTIFACTS": str(tmp_path / "aiperf")}
    command = [sys.executable, str(FAKES / "fake_aiperf.py")]
    scope = write(tmp_path, "not-a-directory", "")
    with pytest.raises(NotADirectoryError):
        Process("AIPerf", command, tmp_path / "log", env, str(scope))
    assert e2.kill_marked(str(scope)) == []  # nothing runs with its mark

    def full(path: Path, *args: object, **kwargs: object) -> int:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(path))

    with monkeypatch.context() as patched:
        patched.setattr(Path, "write_text", full)
        with pytest.raises(OSError, match="No space left on device"):
            Process("AIPerf", command, tmp_path / "log", env, str(tmp_path))
    assert e2.kill_marked(str(tmp_path)) == []


def marked(mark: str) -> subprocess.Popen:
    """A process that lasts a minute, marked as a tool of e2 run with ``mark``."""
    env = os.environ | {e2.TOOL_MARK: mark}
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env)


@PROC
def test_a_marked_process_left_by_an_earlier_run_is_found_and_killed(tmp_path: Path) -> None:
    # A mark is its run's scope, the root its run lock is in, and the run's pid and the tool's start.
    here, beside = tmp_path / "a" / "flatshare", tmp_path / "a" / "flatshare" / "b"
    earlier, other = marked(f"{here}/123-456"), marked(f"{beside}/123-456")
    try:
        assert e2.kill_marked(str(here), f"{here}/123-789") == []
        assert e2.kill_marked(str(here)) == [earlier.pid]  # not a scope inside its root's
        assert earlier.wait(timeout=5) == -9 and other.poll() is None
        assert e2.kill_marked(str(beside), f"{beside}/123-456") == [other.pid]
        assert other.wait(timeout=5) == -9
    finally:
        earlier.kill()
        other.kill()


def wait_for(path: Path, timeout: float = 10.0) -> Path:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)
    return path


def gone(pid: int, timeout: float = 5.0) -> bool:
    """Whether ``pid`` exits within ``timeout``: a killed orphan lasts until init reaps it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


# A point, against fakes


Routes = dict[str, Callable[[int], tuple[int, Any]]]


class Service(ThreadingHTTPServer):
    """An HTTP service on loopback; each route answers from the number of requests it has had."""

    def __init__(self, routes: Routes) -> None:
        self.routes = routes
        self.hits: Counter[str] = Counter()
        super().__init__(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class UnixService(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """The same service on a Unix socket of its own, as the engine through the tunnel, or the gateway's
    socket; its address is ``unix:`` and the socket's path."""

    daemon_threads = True

    def __init__(self, routes: Routes) -> None:
        self.routes = routes
        self.hits: Counter[str] = Counter()
        # A short path: a socket's must fit sun_path, which a test's tmp_path may not.
        self.path = Path(tempfile.mkdtemp(prefix="flatshare-e2-")) / "s.sock"
        super().__init__(str(self.path), _Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"unix:{self.path}"


class CountingEngine(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """An engine's /metrics on a Unix socket, as the tunnel's on cyan, that counts the connections it accepts and
    keeps each one open between requests (HTTP/1.1), as vLLM does for 60 s; the requests numbered in ``drop`` get
    no answer, and lose their connection."""

    daemon_threads = True

    def __init__(self, drop: frozenset[int] = frozenset()) -> None:
        self.accepted = self.requests = 0
        self.drop = drop
        # A short path: a socket's must fit sun_path, which a test's tmp_path may not.
        self.path = Path(tempfile.mkdtemp(prefix="flatshare-kv-")) / "s.sock"
        super().__init__(str(self.path), _KeepAlive)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def get_request(self):
        request = super().get_request()
        self.accepted += 1
        return request


class _KeepAlive(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: CountingEngine

    def do_GET(self) -> None:
        self.server.requests += 1
        if self.server.requests in self.server.drop:
            self.close_connection = True
            return
        body = (
            b'vllm:kv_cache_usage_perc{engine="0"} 0.5\n'
            b'vllm:num_requests_running{engine="0"} 2\n'
            b'vllm:num_requests_waiting{engine="0"} 0\n'
        )
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class _Handler(BaseHTTPRequestHandler):
    server: Service | UnixService

    def answer(self) -> None:
        path = self.path.split("?")[0]
        self.server.hits[path] += 1
        route = self.server.routes.get(path)
        status, body = route(self.server.hits[path]) if route else (404, "")
        data = (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = answer

    def log_message(self, *args: object) -> None:
        pass


def gateway_metrics(n: int, ended: int = 0) -> tuple[int, str]:
    """The gateway's metrics at its ``n``-th scrape, when it had ended ``ended`` requests."""
    return 200, (
        f'flatshare_gateway_request_outcomes_total{{class="realtime",outcome="completed"}} {ended}\n'
        f'flatshare_gateway_requests_total{{class="realtime",status="200"}} {10 * n}\n'
        f'flatshare_gateway_first_byte_seconds_bucket{{class="realtime",le="0.1"}} {8 * n}\n'
        f'flatshare_gateway_first_byte_seconds_bucket{{class="realtime",le="0.5"}} {10 * n}\n'
        f'flatshare_gateway_first_byte_seconds_bucket{{class="realtime",le="+Inf"}} {10 * n}\n'
        f'flatshare_gateway_first_byte_seconds_sum{{class="realtime"}} {0.9 * n}\n'
        f'flatshare_gateway_first_byte_seconds_count{{class="realtime"}} {10 * n}\n'
    )


def engine_metrics(n: int) -> tuple[int, str]:
    return 200, (
        f'vllm:num_preemptions_total{{engine="0"}} {n}\n'
        f'vllm:prompt_tokens_by_source_total{{engine="0",source="local_compute"}} {100 * n}\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="2275"} 1\n'
    )


REVISION = "8591804019c8b22094c3b5b4454e0edc05dffc98"
"""The frozen box's model revision (deploy/profiles/rtx-5060-ti.toml)."""


def fake_engine(scheduler: dict[str, Any]) -> UnixService:
    """vLLM on a GPU node, through its tunnel's socket: ``scheduler`` is what its /server_info reports,
    as variant.sh use would leave it."""
    vllm_config = {
        "scheduler_config": scheduler,
        # The model as vLLM's offline mode names it: by its path in the Hugging Face cache.
        "model_config": {
            "model": f"/home/alice/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots/{REVISION}",
            "revision": REVISION,
            "max_model_len": 32768,
            "hf_token": "hf_secret",
        },
        "kernel_config": {"linear_backend": "auto", "moe_backend": "auto"},
        "additional_config": {},
    }
    return UnixService(
        {
            "/health": lambda n: (200, ""),
            "/metrics": engine_metrics,
            "/server_info": lambda n: (200, {"vllm_config": vllm_config}),
            "/version": lambda n: (200, {"version": "0.32.0"}),
            # Busy at first, as vLLM is while KV transfers finish.
            "/reset_prefix_cache": lambda n: (200, {"success": n > 1}),
        }
    )


@dataclass(frozen=True)
class FakeHost(Host):
    def aiperf(self) -> list[str]:
        return [sys.executable, str(FAKES / "fake_aiperf.py")]

    def bots(self) -> list[str]:
        return [sys.executable, str(FAKES / "fake_bots.py")]


class FakeSystemd(Systemd):
    """Records systemctl calls; the worker runs with fixed endpointing and registers at once. A unit
    in ``failing`` does not start, and its log says why. ``journals`` gives a unit's journal from
    ``since`` to ``until``; otherwise the gateway's holds the request lines of ten realtime requests
    (GATEWAY_FIRST_BYTES) and any other unit's only the worker's registration. ``ended`` counts the
    requests the gateway's journal reads show ending, which the fake gateway's metrics report: a point
    reads its base before the journal and its last count before the check reads it again."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.running: set[str] = set()
        self.failing: set[str] = set()
        self.journals: dict[str, Callable[[float, float | None], str]] = {}
        self.ended = 0
        self.env = env or {
            "FLATSHARE_TURN_DETECTION": "vad",
            "FLATSHARE_LLM_UDS": "/run/user/1/flatshare/gateway.sock",
        }

    def __call__(self, *args: str) -> str:
        if args[0] == "show":
            return " ".join(f"{k}={v}" for k, v in self.env.items())
        self.calls.append(args)
        if args[0] in ("start", "restart") and args[1] in self.failing:
            raise E2Error(f"systemctl --user {' '.join(args)}: Job for {args[1]}.service failed")
        if args[0] == "start":
            self.running.add(args[1])
        elif args[0] == "stop":
            self.running.discard(args[1])
        return ""

    def active(self, unit: str) -> bool:
        return unit in self.running

    def journal(self, unit: str, since: float, until: float | None = None) -> str:
        if unit in self.failing:
            return f"{unit}: starting\n{unit}: config: unknown field `upstream_priority`\n"
        if unit in self.journals:
            text = self.journals[unit](since, until)
        elif unit == "flatshare-gateway":
            text = "\n".join(
                logged_request(since + n / 100, "realtime", first_byte)
                for n, first_byte in enumerate(GATEWAY_FIRST_BYTES, 1)
            )
        else:
            return "registered worker"
        if unit == "flatshare-gateway":
            self.ended += ended_requests(text, since, until)
        return text


def ended_requests(journal: str, since: float, until: float | None) -> int:
    """The requests that went upstream with a class (a ``class`` and an ``outcome``) whose lines in a
    fake gateway journal end within ``since``..``until``."""
    ended = 0
    for line in journal.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("message") == "request" and "class" in record and "outcome" in record:
            at = datetime.fromisoformat(record["timestamp"]).timestamp()
            ended += since < at <= (until or math.inf)
    return ended


GATEWAY_FIRST_BYTES = (50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0)
"""The first bytes of the realtime requests the fake gateway logs during the calls."""


@dataclass
class Cyan:
    host: FakeHost
    systemd: FakeSystemd
    gateway: Service
    gateway_socket: UnixService
    engine: UnixService
    scheduler: dict[str, Any]
    """What the engine's /server_info reports, as variant.sh use would leave it: the frozen box."""
    arms: Path


@pytest.fixture
def cyan(tmp_path: Path) -> Iterator[Cyan]:
    if not hasattr(os, "waitid"):
        pytest.skip(e2.NEEDS_WAITID)  # every point runs its tools
    home = tmp_path / "home"
    config = home / ".config" / "flatshare"
    config.mkdir(parents=True)
    (config / "gateway.toml").write_text(EXAMPLE.read_text())
    (config / "gateway-keys.env").write_text("FLATSHARE_GATEWAY_AGENTS_KEY=agents-key\n")
    (config / "livekit.env").write_text("LIVEKIT_API_KEY=k\nLIVEKIT_API_SECRET=s\n")
    (home / "flatshare" / "data" / "corpus").mkdir(parents=True)
    scheduler = dict(FROZEN_SCHEDULER)
    systemd = FakeSystemd()
    # AIPerf's TCP listener, and the socket from which a point reads the gateway's metrics.
    gateway = Service({"/ready": lambda n: (200, "")})
    socket = UnixService({"/metrics": lambda n: gateway_metrics(n, systemd.ended)})
    engine = fake_engine(scheduler)
    host = FakeHost(
        home=home, gateway_url=gateway.url, gateway_socket=socket.url, engine_url=engine.url
    )
    arms = tmp_path / "arms"
    arms.mkdir()
    yield Cyan(host, systemd, gateway, socket, engine, scheduler, arms)
    for service in (gateway, socket, engine):
        service.shutdown()


def run(cyan: Cyan, name: str, text: str, out: str | None = None) -> dict[str, Any]:
    """A point on the fake cyan, without the KV sampler, whose /metrics reads the fake engine's
    counters would count (test_an_agent_point_samples_the_engines_kv runs it)."""
    arm_file = write(cyan.arms, f"{name}.toml", text)
    path = cyan.host.root / "runs" / (out or name)
    return run_point(arm_file, path, cyan.host, cyan.systemd, kv_sample_s=None)


def run_stock(cyan: Cyan, out: str | None = None) -> dict[str, Any]:
    """The stock arm, on an engine switched to it as variant.sh use would."""
    cyan.scheduler.update(STOCK_SCHEDULER)
    return run(cyan, "a2-stock-s4", ARM, out)


def record(cyan: Cyan, name: str) -> dict[str, Any]:
    return json.loads((cyan.host.root / "runs" / name / "run.json").read_text())


def test_voice_alone_runs_the_calls_against_the_box_as_it_is(cyan: Cyan) -> None:
    summary = run(cyan, "a1-voice", VOICE_ALONE)
    out = cyan.host.root / "runs" / "a1-voice"
    assert cyan.systemd.calls == []  # nothing to change, so nothing restarts
    assert cyan.engine.hits["/reset_prefix_cache"] == 2  # retried until the engine could
    assert not (out / "aiperf.log").exists()
    run_json = record(cyan, "a1-voice")
    assert run_json["status"] == "complete"
    frozen = ["--scheduling-policy", "--max-num-batched-tokens", "--long-prefill-token-threshold"]
    model = ["--model", "--revision", "--max-model-len"]  # the box's profile's
    assert run_json["engine"] == {
        "server": "vllm",
        "version": "0.32.0",
        "pool_tokens": 36400,
        "flags": {"checked": [*frozen, *model], "unchecked": []},
    }
    assert run_json["profile"] == "rtx-5060-ti"
    engine = json.loads((out / "engine.json").read_text())
    assert engine["scheduler_config"] == FROZEN_SCHEDULER
    assert engine["model_config"]["hf_token"] == "<redacted>"
    assert run_json["cyan"]["llm_path"].endswith("gateway.sock")
    assert run_json["versions"]["aiperf"] == "0.13.0"
    # No campaign named the GPU node's revision; it goes beside cyan's.
    assert list(run_json["versions"])[:2] == ["source", "gpu_source"]
    assert run_json["versions"]["gpu_source"] is None
    assert (out / "arm.toml").read_text().startswith("arm = 1")
    args = json.loads((out / "bots.args").read_text())
    assert args[args.index("--turns") + 1] == "25" and args[args.index("--callers") + 1] == "2"
    assert "--save-audio" not in args and run_json["save_audio"] is False
    assert "--warmup-turns" not in args  # an arm without warm-up turns runs the callers as before
    assert summary["name"] == "a1-voice" and summary["background"] is None
    assert summary["voice"]["v2v_ms"]["p95"] == 1800.0
    assert summary["engine"]["preemptions"] == 1  # one scrape apart
    # Exactly from the gateway's request lines of the calls, which the run keeps.
    assert len((out / "gateway.jsonl").read_text().splitlines()) == len(GATEWAY_FIRST_BYTES)
    assert summary["gateway"]["realtime_first_byte_ms"] == {
        "n": 10, "mean": 95.0, "p50": 95.0, "p95": 135.5, "p99": 139.1, "source": "log",
    }  # fmt: skip
    assert summary["worker"] is None  # nor does this worker log turn lines
    assert json.loads((out / "summary.json").read_text()) == summary


@PROC
def test_a_point_kills_what_its_own_runs_left_and_not_another_sessions_tools(cyan: Cyan) -> None:
    # Two test sessions on one host, each with its own home and run lock: a point's start killed
    # every marked process of the user, so one session's point killed the other's tools (lane UP's
    # make check saw them exit -9). It kills only what a run under its own lock left.
    other_root = cyan.host.root.parent / "other-session" / "flatshare"
    left, others = marked(f"{cyan.host.root}/1-1"), marked(f"{other_root}/2-2")
    try:
        run(cyan, "a1-voice", VOICE_ALONE)
        assert left.wait(timeout=5) == -9
        assert others.poll() is None  # the other session's tool runs on
        assert record(cyan, "a1-voice")["leftovers_killed"] == [left.pid]
    finally:
        left.kill()
        others.kill()


@PROC
def test_a_point_ends_the_group_a_killed_runs_tool_left_its_titled_processes_too(
    cyan: Cyan, tmp_path: Path
) -> None:
    """A run killed before it could stop AIPerf left it running, its service titled, which the marks
    alone do not find: the next point ends them by the group named in the scope."""
    pytest.importorskip("setproctitle")
    artifacts = tmp_path / "left"
    env = os.environ | {"FLATSHARE_BENCH_ARTIFACTS": str(artifacts), "FAKE_AIPERF": "titled"}
    command = [sys.executable, str(FAKES / "fake_aiperf.py")]
    tool = Process("AIPerf", command, tmp_path / "left.log", env, e2.tool_scope(cyan.host))
    child = int(wait_for(artifacts / "child.pid").read_text())
    try:
        titled(child)
        run(cyan, "a1-voice", VOICE_ALONE)
        killed = record(cyan, "a1-voice")["leftovers_killed"]
        assert sorted(killed) == sorted([tool.popen.pid, child])
        assert gone(child) and tool.popen.wait(timeout=5) == -9
    finally:
        for pid in (child, tool.popen.pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_a_point_whose_leftovers_cannot_be_ended_fails_and_says_why(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(scope: str) -> list[int]:
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), scope)

    monkeypatch.setattr(e2, "kill_left", refused)
    with pytest.raises(PermissionError):
        run(cyan, "a1-voice", VOICE_ALONE)
    run_json = record(cyan, "a1-voice")
    assert run_json["status"] == "failed" and "Permission denied" in run_json["error"]


def test_the_generated_configuration_is_private_from_its_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # gateway-e2.toml was written with the umask's mode and made mode 600 after, so that it was readable in between
    # (644 under 022, 666 under 0), and the staged file an interrupted write left was written over in place: whoever
    # had opened it read the next configuration's key digests.
    path, staged = tmp_path / "gateway-e2.toml", tmp_path / "gateway-e2.toml.new"
    staged.write_text("stale\n")
    staged.chmod(0o644)
    monkeypatch.setattr(Path, "chmod", lambda *args, **kwargs: None)  # no mode set after the fact
    mask = os.umask(0)
    try:
        with staged.open() as reader:
            assert e2._write_private(path, "digests\n")
            assert reader.read() == "stale\n"
    finally:
        os.umask(mask)
    assert path.read_text() == "digests\n" and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not staged.exists()
    assert not e2._write_private(path, "digests\n"), "an unchanged configuration is not written"


def test_a_point_with_agents_configures_cyan_and_stops_aiperf_after_the_calls(cyan: Cyan) -> None:
    summary = run_stock(cyan)
    out = cyan.host.root / "runs" / "a2-stock-s4"
    units = cyan.host.units
    gateway_link = units / "flatshare-gateway.service.d" / "variant-e2-gateway.conf"
    worker = units / "flatshare-voice-worker.service.d"
    assert gateway_link.readlink().parts[-2:] == ("e2-gateway", "flatshare-gateway.conf")
    assert {link.readlink().parts[-2] for link in worker.glob("variant-*.conf")} == {
        "no-preemptive-generation",
        "no-events",
    }
    assert (
        REPO / "deploy" / "cyan" / "variants" / "no-events" / "flatshare-voice-worker.conf"
    ).exists()
    assert cyan.systemd.calls == [  # the worker restarts once for both of its drop-ins
        ("daemon-reload",),
        ("restart", "flatshare-gateway"),
        ("restart", "flatshare-voice-worker"),
    ]
    generated = cyan.host.config / "gateway-e2.toml"
    assert stat.S_IMODE(generated.stat().st_mode) == 0o600  # it holds the key digests
    config = tomllib.loads(generated.read_text())
    assert config["upstream_priority"] == [] and config["admission"]["policy"] == "static-cap"
    assert config["tenants"]["agents"]["keys_sha256"] == ["<sha256 of the agents key>"]
    env = json.loads((out / "aiperf" / "env.json").read_text())
    assert env["FLATSHARE_BENCH_URL"] == f"{cyan.gateway.url}/v1/chat/completions"
    assert env["FLATSHARE_BENCH_API_KEY"] == "agents-key"
    assert env["FLATSHARE_BENCH_SESSIONS"] == "4"
    # The model the engine serves, by the box's profile: the arm's flags name none.
    assert (
        env["FLATSHARE_BENCH_MODEL"]
        == env["FLATSHARE_BENCH_TOKENIZER"]
        == "Qwen/Qwen3-4B-Instruct-2507-FP8"
    )
    assert env["AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID"] == "false"  # from configs/aiperf.env
    run_json = record(cyan, "a2-stock-s4")
    assert run_json["aiperf"] == {"exit_status": 0}
    assert run_json["cyan"]["worker"] == {
        "restarted": True,
        "preemptive_generation": False,
        "events": False,
    }
    assert "agents-key" not in (out / "run.json").read_text()
    assert run_json["cyan"]["gateway"]["config"]["tenants"]["agents"]["keys_sha256"] == [
        "<redacted>"
    ]
    assert "Exporting all records" in (out / "aiperf.log").read_text()  # SIGINT let it finish
    background = summary["background"]
    assert background["requests"] > 0 and background["errors"] == 0
    assert background["tokens_per_s"] == pytest.approx(
        background["requests_per_s"] * 1010, rel=0.01
    )
    assert 100 <= background["ttft_ms"]["p50"] <= 109
    assert 20 <= background["itl_ms"]["p50"] <= 24  # AIPerf's per request
    window = run_json["window"]
    assert 0.3 <= window["end"] - window["start"] < 5


def test_an_agent_point_samples_the_engines_kv(cyan: Cyan) -> None:
    # The engine's pool full on every other /metrics read (0.999 of 2,274 blocks in use: 2 free, under
    # two calls' 128) and half used on the others; its counters stand still.
    def metrics(n: int) -> tuple[int, str]:
        _, counters = engine_metrics(1)
        full = n % 2 == 1
        return 200, counters + (
            f'vllm:kv_cache_usage_perc{{engine="0"}} {0.999 if full else 0.5}\n'
            f'vllm:num_requests_running{{engine="0"}} {4 if full else 2}\n'
            'vllm:num_requests_waiting{engine="0"} 1\n'
        )

    model = {
        "model": "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "revision": REVISION,
        "max_model_len": 32768,
    }
    config = {
        "model_config": model,
        "cache_config": {"block_size": 16, "num_gpu_blocks": 2275},
        "scheduler_config": cyan.scheduler,
        "additional_config": {},
    }
    cyan.engine.routes["/metrics"] = metrics
    cyan.engine.routes["/server_info"] = lambda n: (200, {"vllm_config": config})
    cyan.scheduler.update(STOCK_SCHEDULER)
    arm_file = write(cyan.arms, "a2-stock-s4.toml", ARM)
    out = cyan.host.root / "runs" / "a2-stock-s4"
    summary = run_point(arm_file, out, cyan.host, cyan.systemd, kv_sample_s=0.02)
    assert record(cyan, "a2-stock-s4")["kv_sample_s"] == 0.02
    lines = (out / "metrics" / "engine-kv.csv").read_text().splitlines()
    assert lines[0] == "time,kv_cache_usage,running,waiting"
    window = record(cyan, "a2-stock-s4")["window"]
    times = [float(line.split(",")[0]) for line in lines[1:]]
    assert times == sorted(times) and window["start"] <= times[0]
    kv = summary["kv"]
    assert kv["samples"] >= 5 and 0.25 <= kv["running_share"] <= 0.75
    assert kv["free_blocks"]["min"] == 2 and kv["running"]["max"] == 4
    assert kv["waiting"] == {"p50": 1, "max": 1}
    # A voice-alone point has no agents, and so no samples.
    cyan.scheduler.update(FROZEN_SCHEDULER)
    alone = run_point(
        write(cyan.arms, "a1-voice.toml", VOICE_ALONE),
        cyan.host.root / "runs" / "a1-voice",
        cyan.host,
        cyan.systemd,
        kv_sample_s=0.02,
    )
    assert (
        alone["kv"] is None
        and not (cyan.host.root / "runs" / "a1-voice" / "metrics" / "engine-kv.csv").exists()
    )
    assert record(cyan, "a1-voice")["kv_sample_s"] is None
    # An interval of 0 samples nothing, as W5's points ran.
    cyan.scheduler.update(STOCK_SCHEDULER)
    unsampled = cyan.host.root / "runs" / "a2-stock-s4-w5"
    plain = run_point(
        write(cyan.arms, "a2-stock-s4-w5.toml", ARM),
        unsampled,
        cyan.host,
        cyan.systemd,
        kv_sample_s=0,
    )
    assert plain["kv"] is None and not (unsampled / "metrics" / "engine-kv.csv").exists()
    assert record(cyan, "a2-stock-s4-w5")["kv_sample_s"] is None


def test_the_kv_sampler_keeps_one_connection_to_the_engine(tmp_path: Path) -> None:
    # Through cyan's tunnel each new connection is a new SSH session on the GPU node: the sampler holds one for
    # its run, and opens another only after a request failed (the 4th, which loses its own).
    engine = CountingEngine(drop=frozenset({4}))
    sampler = e2._KvSampler(f"unix:{engine.path}", tmp_path, 0.01)
    deadline = time.monotonic() + 20
    while engine.requests < 10 and time.monotonic() < deadline:
        time.sleep(0.01)
    sampler.stop()
    engine.shutdown()
    assert engine.requests >= 10 and engine.accepted == 2
    rows = (tmp_path / "engine-kv.csv").read_text().splitlines()[1:]
    assert len(rows) == engine.requests - 1  # every answered sample, the failed one left out


def test_e2_run_takes_the_kv_samplers_interval(tmp_path: Path, monkeypatch) -> None:
    seen: list[float] = []

    def recorded(arm_file, out, host, **options):
        seen.append(options["kv_sample_s"])
        raise E2Error("recorded")

    monkeypatch.setattr(e2, "run_point", recorded)
    monkeypatch.setenv("HOME", str(tmp_path))  # Host.load reads the host's settings from the home
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGHUP, signal.SIGTERM)}
    arm = write(tmp_path, "a2-stock-s4.toml", ARM)
    try:
        for extra in ([], ["--kv-sample-s", "0"], ["--kv-sample-s", "2.5"]):
            assert main(["e2", "run", str(arm), "--out", str(tmp_path / "out"), *extra]) == 1
        with pytest.raises(SystemExit):
            main(["e2", "run", str(arm), "--out", str(tmp_path / "out"), "--kv-sample-s", "-1"])
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    assert seen == [0.5, 0.0, 2.5]


def test_a_campaign_sets_the_agents_seed_and_run_json_keeps_it(cyan: Cyan) -> None:
    arm_file = write(cyan.arms, "a2-stock-s4.toml", ARM)
    cyan.scheduler.update(STOCK_SCHEDULER)
    runs = cyan.host.root / "runs"
    for out, seed, used in (("own", None, 11), ("campaign", 13, 13)):
        run_point(arm_file, runs / out, cyan.host, cyan.systemd, agents_seed=seed)
        run_json = json.loads((runs / out / "run.json").read_text())
        env = json.loads((runs / out / "aiperf" / "env.json").read_text())
        assert (env["FLATSHARE_BENCH_SEED"], run_json["agents_seed"]) == (str(used), used)
        assert run_json["arm"]["agents"]["seed"] == 11  # the arm as read
    cyan.scheduler.update(FROZEN_SCHEDULER)
    run(cyan, "a1-voice", VOICE_ALONE)  # no replay, no seed
    assert record(cyan, "a1-voice")["agents_seed"] is None
    with pytest.raises(E2Error, match="the agents' seed cannot be negative"):
        run_point(arm_file, runs / "negative", cyan.host, cyan.systemd, agents_seed=-1)


def stale_metadata(monkeypatch: pytest.MonkeyPatch) -> str:
    """flatshare-bench's installed metadata as an editable install left it before a version change; returns
    the version of the source it runs."""
    versions = {"flatshare-bench": "0.1.0"}
    real = e2.metadata.version
    monkeypatch.setattr(e2.metadata, "version", lambda name: versions.get(name) or real(name))
    return tomllib.loads(e2.PYPROJECT.read_text())["project"]["version"]


def test_an_install_older_than_its_source_is_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # This test's own venv is synced from the source it runs.
    assert e2._stale_install() is None
    source = stale_metadata(monkeypatch)
    assert e2._stale_install() == (
        f"flatshare-bench's venv was installed at version 0.1.0 and its source is {source}: sync the venv "
        "(deploy/cyan/bootstrap.sh) so that run.json names the version that runs"
    )
    # An installed wheel has no pyproject.toml beside it: its metadata is its source's.
    assert e2._stale_install(tmp_path / "pyproject.toml") is None


def test_a_point_on_an_install_older_than_its_source_is_refused_before_it_starts(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm_file = write(cyan.arms, "a2-stock-s4.toml", ARM)
    source = stale_metadata(monkeypatch)
    out = cyan.host.root / "runs" / "stale"
    with pytest.raises(E2Error, match=re.escape(f"version 0.1.0 and its source is {source}")):
        run_point(arm_file, out, cyan.host, cyan.systemd)
    assert not out.exists() and cyan.systemd.calls == []  # nothing made, nothing switched


def test_the_same_arm_again_restarts_nothing(cyan: Cyan) -> None:
    run_stock(cyan)
    cyan.systemd.calls.clear()
    run_stock(cyan, out="again")
    assert cyan.systemd.calls == []
    cyan.scheduler.update(FROZEN_SCHEDULER)
    run(cyan, "a1-voice", VOICE_ALONE)  # and back: the drop-ins go, the units restart
    assert cyan.systemd.calls == [
        ("daemon-reload",),
        ("restart", "flatshare-gateway"),
        ("restart", "flatshare-voice-worker"),
    ]
    assert not any(cyan.host.units.glob("*.service.d/variant-*.conf"))


def logged_request(
    at: float, slo_class: str | None, first_byte_ms: float | None, **fields: Any
) -> str:
    """One of the gateway's JSON request lines (log_format = "json"), for a request that ended at
    ``at``; like the gateway, it leaves out the fields it has no value for."""
    stamp = (
        datetime.fromtimestamp(at, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    line = {
        "timestamp": stamp,
        "level": "INFO",
        "message": "request",
        "class": slo_class,
        "route": "chat_completions",
        "status": 200,
        "outcome": "completed",
        "first_byte_ms": first_byte_ms,
        "total_ms": 2000.0,
        "target": "flatshare_gateway::request",
    }
    return json.dumps({k: v for k, v in (line | fields).items() if v is not None})


def worker_line(name: str, message: str, **fields: Any) -> str:
    """One of the voice worker's JSON log lines, as LiveKit's formatter writes them."""
    line = {"message": message, "level": "INFO", "name": name, **fields, "pid": 7}
    return json.dumps(line | {"timestamp": "2026-10-23T05:00:00+00:00"})


TURNS, SESSIONS = "flatshare_voice.turns", "flatshare_voice.events"


def test_a_point_keeps_the_gateways_requests_and_the_workers_lines(cyan: Cyan) -> None:
    def gateway(since: float, until: float | None) -> str:
        assert until is not None and since < until  # the window of the calls
        first_bytes = (120.0, 180.0, 240.0)
        return "\n".join(
            [
                logged_request(since - 0.5, "realtime", 999.0),  # in journalctl's second, earlier
                *(
                    logged_request(since + n / 100, "realtime", b)
                    for n, b in enumerate(first_bytes, 1)
                ),
                logged_request(since + 0.05, "background", 3000.0),
                logged_request(since + 0.06, None, None, route="events", status=202, outcome=None),
                '{"timestamp":"2026-10-23T05:00:00.000000Z","message":"lifecycle event"}',
                "flatshare-gateway serving",
            ]
        )

    reads: list[float | None] = []
    counts = {"turns": 1, "events_sent": 10, "events_dropped": 1, "events_failed": 0}
    ended = [worker_line(SESSIONS, "session ended", session_id=f"s{n}", **counts) for n in (0, 1)]

    def worker(since: float, until: float | None) -> str:
        reads.append(until)
        return "\n".join(
            [
                "registered worker",
                worker_line(TURNS, "turn", session_id="s0", turn=0, e2e_ms=1000.0),
                worker_line(
                    TURNS, "turn", session_id="s1", turn=0, llm_ttft_ms=50.0, e2e_ms=1200.0
                ),
                *(ended if len(reads) >= 3 else []),  # the sessions end a little after the calls
            ]
        )

    cyan.systemd.journals = {"flatshare-gateway": gateway, "flatshare-voice-worker": worker}
    summary = run(cyan, "a1-voice", VOICE_ALONE)
    out = cyan.host.root / "runs" / "a1-voice"
    kept = [json.loads(line) for line in (out / "gateway.jsonl").read_text().splitlines()]
    assert len(kept) == 6 and {line["message"] for line in kept} == {"request"}
    # Exact, from the three realtime requests that ended during the calls.
    assert summary["gateway"]["realtime_first_byte_ms"] == {
        "n": 3, "mean": 180.0, "p50": 180.0, "p95": 234.0, "p99": 238.8, "source": "log",
    }  # fmt: skip
    assert summary["gateway"]["background_first_byte_ms"]["p50"] == 3000.0
    assert len(reads) == 3, "it waited for both calls' sessions to end"
    assert "session ended" in (out / "voice-worker.log").read_text()
    worker_summary = summary["worker"]
    assert (worker_summary["sessions"], worker_summary["turns"]) == (2, 2)
    assert worker_summary["events"] == {"sent": 20, "dropped": 2, "failed": 0}
    assert worker_summary["waterfall"]["e2e_ms"]["p50"] == 1100.0
    assert worker_summary["waterfall"]["llm_ttft_ms"]["n"] == 1
    assert worker_summary["waterfall"]["end_of_turn_ms"] == {"n": 0}


def test_a_call_opens_with_its_warm_up_turns_which_nothing_measures(cyan: Cyan) -> None:
    def worker(since: float, until: float | None) -> str:
        lines = ["registered worker"]
        for session in ("s0", "s1"):  # each session's reply to the warm-up line comes first
            for turn, e2e in ((0, 9000.0), (1, 1000.0), (2, 1200.0), (3, 1400.0)):
                lines.append(worker_line(TURNS, "turn", session_id=session, turn=turn, e2e_ms=e2e))
            lines.append(worker_line(SESSIONS, "session ended", session_id=session))
        return "\n".join(lines)

    cyan.systemd.journals = {"flatshare-voice-worker": worker}
    arm = VOICE_ALONE + "[voice]\ncallers = 2\nturns = 3\nwarmup_turns = 1\n"
    summary = run(cyan, "a1-warm", arm)
    out = cyan.host.root / "runs" / "a1-warm"
    args = json.loads((out / "bots.args").read_text())
    assert args[args.index("--warmup-turns") + 1] == "1"
    assert record(cyan, "a1-warm")["arm"]["voice"]["warmup_turns"] == 1
    turns = [json.loads(line) for line in (out / "bots" / "turns.jsonl").read_text().splitlines()]
    assert [t["turn"] for t in turns if t.get("warmup")] == [0, 0]  # one a call, kept, marked
    waterfall = summary["worker"]["waterfall"]["e2e_ms"]
    assert (summary["worker"]["turns"], waterfall["n"], waterfall["p50"]) == (6, 6, 1200.0)
    assert "calls: 2 callers, 3 turns each after 1 warm-up" in (out / "run.log").read_text()


def test_a_point_checks_that_the_journal_kept_every_request_line(cyan: Cyan) -> None:
    run(cyan, "a1-voice", VOICE_ALONE)
    # Ten requests ended between the idle moments before and after the point, and the journal has their lines.
    assert record(cyan, "a1-voice")["gateway_log"] == {"request_lines": 10, "requests_ended": 10}

    # journald dropped a line: what the check reads lacks one of the requests the gateway counted.
    reads = itertools.count()

    def dropping(since: float, until: float | None) -> str:
        n = 10 if next(reads) == 0 else 9
        return "\n".join(
            logged_request(since + i / 100, "realtime", 100.0) for i in range(1, n + 1)
        )

    cyan.systemd.journals["flatshare-gateway"] = dropping
    with pytest.raises(E2Error, match="kept 9 request lines of the 10 requests it ended"):
        run(cyan, "a1-voice", VOICE_ALONE, out="dropped")
    assert record(cyan, "dropped")["status"] == "failed"


def test_a_gateway_that_does_not_go_idle_fails_the_point(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(e2, "GATEWAY_IDLE_S", 0.2)
    cyan.gateway_socket.routes["/metrics"] = lambda n: (
        200,
        f'flatshare_gateway_request_outcomes_total{{class="background",outcome="completed"}} {n}\n'
        'flatshare_gateway_streams_active{class="background"} 1\n',
    )
    with pytest.raises(E2Error, match="the gateway did not go idle within"):
        run(cyan, "a1-voice", VOICE_ALONE)


def test_a_gateway_that_logs_no_request_lines_fails_the_point(cyan: Cyan) -> None:
    cyan.systemd.journals = {
        "flatshare-gateway": lambda since, until: "flatshare-gateway serving"
    }  # text
    with pytest.raises(E2Error, match="the gateway logged no request line for the calls"):
        run(cyan, "a1-voice", VOICE_ALONE)
    run_json = record(cyan, "a1-voice")
    assert run_json["status"] == "failed" and 'log_format = "json"' in run_json["error"]
    assert not (cyan.host.root / "runs" / "a1-voice" / "summary.json").exists()


def test_the_worker_journal_waits_only_for_sessions_that_end(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(e2, "SESSIONS_END_S", 0.2)
    reads: list[float | None] = []

    def journal(*lines: str) -> Callable[[float, float | None], str]:
        def read(since: float, until: float | None) -> str:
            reads.append(until)
            return "\n".join(["registered worker", *lines])

        return read

    # A worker that logs no turn lines (a build from before them): nothing to wait for.
    cyan.systemd.journals = {"flatshare-voice-worker": journal()}
    assert run(cyan, "a1-voice", VOICE_ALONE)["worker"] is None
    assert len(reads) == 1
    # Two calls and one session that ends: the wait is bounded, and the summary says what came.
    turn = worker_line(TURNS, "turn", session_id="s", turn=0, e2e_ms=900.0)
    cyan.systemd.journals = {
        "flatshare-voice-worker": journal(
            turn, worker_line(SESSIONS, "session ended", session_id="s")
        )
    }
    reads.clear()
    summary = run(cyan, "a1-voice", VOICE_ALONE, out="again")
    assert len(reads) >= 3
    assert (summary["worker"]["sessions"], summary["worker"]["events"]) == (1, None)


def test_run_json_keeps_each_gpu_nodes_deployed_revision(cyan: Cyan, capsys) -> None:
    arm_file = write(cyan.arms, "a1-voice.toml", VOICE_ALONE)
    out = cyan.host.root / "runs" / "a1-voice"
    sources = gpu_sources(["phastform=v0.3-30-g6851e13", "gray="])
    assert sources == {"phastform": "v0.3-30-g6851e13", "gray": None}
    run_point(arm_file, out, cyan.host, cyan.systemd, gpu_source=sources)
    assert record(cyan, "a1-voice")["versions"]["gpu_source"] == sources
    for values, error in (
        (["phastform"], "'phastform' is not NODE=REVISION"),
        (["phastform=v0.3 dirty"], "is not NODE=REVISION"),
        (["a b=v0.3"], "is not NODE=REVISION"),
        (["gray=a", "gray=b"], "names gray twice"),
    ):
        with pytest.raises(E2Error, match=re.escape(error)):
            gpu_sources(values)
    # The command line refuses one before it measures anything.
    args = ["e2", "run", str(arm_file), "--out", str(out.with_name("again"))]
    assert main([*args, "--gpu-source", "phastform"]) == 1
    assert "--gpu-source 'phastform' is not NODE=REVISION" in capsys.readouterr().err
    assert not out.with_name("again").exists()


def test_the_callers_save_their_audio_when_the_run_asks(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The callers write each call's audio after their calls end, and the point's window and its last
    # metrics end with the calls, not with the writing (it took the window's end after it).
    monkeypatch.setenv("FAKE_BOTS_SAVE_SECONDS", "1.0")
    arm_file = write(cyan.arms, "a1-voice.toml", VOICE_ALONE)
    out = cyan.host.root / "runs" / "a1-voice"
    run_point(arm_file, out, cyan.host, cyan.systemd, save_audio=True)
    assert "--save-audio" in json.loads((out / "bots.args").read_text())
    assert record(cyan, "a1-voice")["save_audio"] is True  # a run option, not the arm's
    saved = float((out / "bots" / "saved").read_text())
    assert record(cyan, "a1-voice")["window"]["end"] < saved - 0.5
    assert (out / "metrics" / "gateway-end.prom").stat().st_mtime < saved - 0.5


def test_smg_runs_only_for_the_points_that_route_through_it(cyan: Cyan) -> None:
    run(cyan, "a8b", 'arm = "8b"\nlabel = "SMG"\nrouter = "smg"\n')
    assert (
        "start",
        "flatshare-smg",
    ) in cyan.systemd.calls and "flatshare-smg" in cyan.systemd.running
    run(cyan, "a1-voice", VOICE_ALONE)
    assert "flatshare-smg" not in cyan.systemd.running


def test_reset_puts_cyan_back(cyan: Cyan) -> None:
    run_stock(cyan)
    cyan.systemd.running.add("flatshare-smg")
    cyan.systemd.calls.clear()
    applied = reset_cyan(cyan.host, cyan.systemd)
    assert applied["gateway"]["restarted"] and applied["worker"]["restarted"]
    assert ("stop", "flatshare-smg") in cyan.systemd.calls
    assert not any(cyan.host.units.glob("*.service.d/variant-*.conf"))


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("exit-early", "AIPerf exited before its profiling phase (exit status 3)"),
        ("exit-later", "AIPerf exited while the calls ran, so its load did not cover them"),
    ],
)
def test_aiperf_failing_fails_the_point_and_stops_everything(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch, mode: str, error: str
) -> None:
    monkeypatch.setenv("FAKE_AIPERF", mode)
    monkeypatch.setenv("FAKE_BOTS_SECONDS", "5")
    with pytest.raises(E2Error, match=re.escape(error)):
        run_stock(cyan)
    run_json = record(cyan, "a2-stock-s4")
    assert run_json["status"] == "failed" and error in run_json["error"]
    assert "failed: AIPerf" in (cyan.host.root / "runs" / "a2-stock-s4" / "run.log").read_text()
    bots_pid = cyan.host.root / "runs" / "a2-stock-s4" / "bots.pid"
    if bots_pid.exists():  # the calls had started: they were stopped too
        assert gone(int(bots_pid.read_text()))


def test_callers_crashing_fail_the_point_and_stop_aiperf(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_BOTS", "crash")
    with pytest.raises(E2Error, match=r"flatshare-voice bots failed \(exit status 1\)"):
        run_stock(cyan)
    out = cyan.host.root / "runs" / "a2-stock-s4"
    assert "Exporting all records" in (out / "aiperf.log").read_text()  # AIPerf got its SIGINT
    assert record(cyan, "a2-stock-s4")["status"] == "failed"


def test_a_wrong_engine_fails_before_anything_starts(cyan: Cyan) -> None:
    cyan.scheduler.update(STOCK_SCHEDULER)
    error = (
        "the engine runs max_num_batched_tokens = 2048, the point needs --max-num-batched-tokens"
        " 1024: switch it with deploy/phastform/variant.sh use frozen --scheduling-policy fcfs"
        " --max-num-batched-tokens 1024 --long-prefill-token-threshold 0"
    )
    with pytest.raises(E2Error, match=re.escape(error)):
        run(cyan, "a4", ARM.replace('"2048"', '"1024"'))
    out = cyan.host.root / "runs" / "a4"
    assert cyan.systemd.calls == [] and not (out / "aiperf.log").exists()
    run_json = record(cyan, "a4")
    assert run_json["status"] == "failed" and run_json["engine"]["server"] == "vllm"
    assert (out / "engine.json").exists()


@pytest.mark.parametrize("route", ["/server_info", "/reset_prefix_cache"])
def test_an_engine_without_the_measurement_drop_in_fails_the_point_and_says_so(
    cyan: Cyan, route: str
) -> None:
    # A production box: vLLM without its development mode has neither route.
    del cyan.engine.routes[route]
    with pytest.raises(E2Error, match=f"no {route}.*variant.sh measure on"):
        run(cyan, "a1-voice", VOICE_ALONE)
    assert record(cyan, "a1-voice")["status"] == "failed"


def test_a_point_on_make_room_fails_on_the_frozen_box(cyan: Cyan) -> None:
    error = (
        f"the engine runs scheduler_cls = None, the point needs --scheduler-cls {MAKE_ROOM_CLS}: "
        "switch it with deploy/phastform/variant.sh use make-room"
    )
    with pytest.raises(E2Error, match=re.escape(error)):
        run(cyan, "a9", 'arm = 9\nlabel = "x"\n[engine]\nvariant = "make-room"\n')
    assert cyan.systemd.calls == [] and record(cyan, "a9")["status"] == "failed"


def test_a_gateway_that_refuses_its_configuration_says_why(cyan: Cyan) -> None:
    cyan.systemd.failing.add("flatshare-gateway")
    with pytest.raises(E2Error, match="unknown field `upstream_priority`") as failed:
        run_stock(cyan)
    assert "Job for flatshare-gateway.service failed" in str(failed.value)
    assert "flatshare-bench e2 reset puts it back on gateway.toml" in str(failed.value)
    run_json = record(cyan, "a2-stock-s4")
    assert run_json["status"] == "failed" and "unknown field" in run_json["error"]
    assert not (cyan.host.root / "runs" / "a2-stock-s4" / "aiperf.log").exists()


def test_the_window_needs_fixed_endpointing(cyan: Cyan) -> None:
    cyan.systemd.env = {"FLATSHARE_LLM_UDS": "/run/user/1/flatshare/gateway.sock"}
    with pytest.raises(E2Error, match="link deploy/cyan/variants/vad-turns"):
        run(cyan, "a1-voice", VOICE_ALONE)


def test_a_complete_point_is_kept_and_an_incomplete_one_moved_aside(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_BOTS", "crash")
    with pytest.raises(E2Error):
        run(cyan, "a1-voice", VOICE_ALONE)
    monkeypatch.delenv("FAKE_BOTS")
    run(cyan, "a1-voice", VOICE_ALONE)
    runs = cyan.host.root / "runs"
    [earlier] = runs.glob("a1-voice.incomplete-*")
    assert json.loads((earlier / "run.json").read_text())["status"] == "failed"
    with pytest.raises(E2Error, match="holds a complete point"):
        run(cyan, "a1-voice", VOICE_ALONE)
    (runs / "other").mkdir()
    (runs / "other" / "notes.txt").write_text("mine")
    with pytest.raises(E2Error, match="is not empty and holds no point"):
        run(cyan, "a1-voice", VOICE_ALONE, out="other")


def test_one_point_at_a_time(cyan: Cyan) -> None:
    lock = cyan.host.root / "runs" / ".flatshare-bench-e2.lock"
    with e2.exclusive(cyan.host):
        with pytest.raises(E2Error, match="another flatshare-bench e2 run"):
            run(cyan, "a1-voice", VOICE_ALONE)
        assert lock.read_text() == f"{os.getpid()}\n"  # the holder's, kept through the refusal


def test_the_engine_command_prints_what_variant_sh_use_takes(tmp_path: Path, capsys) -> None:
    path = write(tmp_path, "a7a.toml", 'arm = "7a"\nlabel = "x"\n[engine]\nvariant = "pr54530"\n')
    assert main(["e2", "engine", str(path)]) == 0
    assert capsys.readouterr().out == "pr54530\n"
    assert main(["e2", "engine", str(write(tmp_path, "bad.toml", "arm = 2\n"))]) == 1
    assert "label is required" in capsys.readouterr().err


def test_the_record_keeps_a_failed_step_even_when_the_engine_is_down(cyan: Cyan) -> None:
    host = replace(cyan.host, engine_url="unix:/nonexistent/engine.sock")  # nothing listens there
    with pytest.raises(E2Error, match="/server_info"):
        run_point(
            write(cyan.arms, "a1.toml", VOICE_ALONE), host.root / "runs" / "a1", host, cyan.systemd
        )
    assert record(cyan, "a1")["status"] == "failed"
