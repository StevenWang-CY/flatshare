"""The serving variant's scheduler (src/flatshare_vllm/serve) on vLLM's real v1 scheduler, on CPU: it schedules as
make-room's AsyncScheduler, step for step, and writes the configuration it was built with to the file
FLATSHARE_ENGINE_EFFECTIVE names, in the format deploy/bin/serving-check reads: every setting the box's serving.json
names is in it, a profile of the scheduler's own values passes the check, and one that differs fails it (the
negative control). Like tests/test_make_room.py, it needs vLLM with torch and skips without them."""

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.core.sched.scheduler")

import test_make_room as T
import test_trace as TR
from vllm.config import KVTransferConfig

REPO = Path(__file__).resolve().parents[3]
SERVE = REPO / "python/flatshare_vllm/src/flatshare_vllm/serve/flatshare_make_room_serve.py"
CHECK = REPO / "deploy/bin/serving-check"
BOX = json.loads((REPO / "deploy/phastform/serving.json").read_text())
"""The box's serving profile, rendered (deploy/profiles/rtx-5060-ti-serve.toml)."""
SWITCHES = BOX["engine"]["make_room"]
"""The switches it serves: 2 preemptions a step, the budget cap with a 48-token window, no reserve, the agents'
reservations taken and their decodes held from a call's first 48 tokens, the calls' prefixes retained within 3,584
tokens for 30 s, and a 400 ms hold for an announced turn."""
HELPERS = ("flatshare_copy", "flatshare_jit", "flatshare_queue")
"""The modules the serve module installs from."""
SUITE_SWITCHES = SWITCHES | {"retain_tokens": 8 * T.BLOCK}
"""The same on the suite's pools of 48 and 64 blocks, which hold a request of max_model_len (24 blocks) beside a
retention of 8 blocks, not of the box's 224."""


@pytest.fixture(scope="session")
def modules(tmp_path_factory):
    """make-room's module, as the variant builds it, under the name the serve module imports, and the serve module."""
    build = TR.load(TR.PACKAGE / "build.py", "flatshare_serve_build")
    path = build.build("make-room", tmp_path_factory.mktemp("make-room"), source_tree=True)
    make_room = TR.load(path, "flatshare_make_room_scheduler")
    for helper in HELPERS:  # beside it on the engine's PYTHONPATH
        TR.load(SERVE.parent / f"{helper}.py", helper)
    return make_room, TR.load(SERVE, "flatshare_make_room_serve")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


JIT = BOX["engine"]["jit"]
"""Just-in-time scheduling as the box serves it: a 30 ms margin over the card's step-time model."""


@pytest.fixture(autouse=True)
def installed(modules, monkeypatch) -> list[tuple]:
    """What the serve scheduler installs on vLLM's engine core, recorded instead: the suite's schedulers run without
    one, and installing would patch the class for every test after."""
    calls: list[tuple] = []
    monkeypatch.setattr(modules[1].flatshare_jit, "install", lambda *args: calls.append(args))
    return calls


def scheduler(cls, model, num_blocks=48, jit=JIT):
    return T.scheduler(
        cls,
        model,
        num_blocks,
        dict(SUITE_SWITCHES),
        max_num_batched_tokens=128,
        long_prefill_token_threshold=64,
        prefix_caching=True,
        additional={"flatshare_jit": jit},
    )


@pytest.mark.parametrize("num_blocks", [48, 64])
@pytest.mark.parametrize("workload", ["suite", "pressure"])
def test_it_schedules_as_make_rooms_async_scheduler(modules, model, workload, num_blocks):
    make_room, serve = modules
    # The class defines its constructor and schedule(), which fixes flatshare_queue's order where the profile installs
    # it and is make-room's otherwise: every other method is make-room's. Without flatshare_queue and flatshare_copy,
    # as here, it schedules as make-room's step for step.
    assert serve.AsyncScheduler.__mro__[1] is make_room.AsyncScheduler
    defined = [
        name
        for name, value in vars(serve.AsyncScheduler).items()
        if callable(value) or isinstance(value, staticmethod | classmethod | property)
    ]
    assert defined == ["__init__", "schedule"]
    requests = {"suite": T.workload, "pressure": TR.pressure_workload}[workload]
    seen = [
        TR.trace(scheduler(cls, model, num_blocks), requests(), sync=False)
        for cls in (make_room.AsyncScheduler, serve.AsyncScheduler)
    ]
    first = next((i for i, (a, b) in enumerate(zip(*seen, strict=True)) if a != b), None)
    assert first is None, (
        f"first divergence at step {first}: {seen[0][first]} against {seen[1][first]}"
    )


def test_without_the_variable_it_writes_nothing(modules, model, monkeypatch):
    serve = modules[1]
    monkeypatch.delenv("FLATSHARE_ENGINE_EFFECTIVE", raising=False)
    monkeypatch.setattr(serve, "write", lambda *args: pytest.fail("wrote the configuration"))
    scheduler(serve.AsyncScheduler, model)


def leaf_paths(table: dict, prefix: tuple = ()) -> list[tuple]:
    return [
        path
        for key, value in table.items()
        for path in (
            leaf_paths(value, (*prefix, key)) if isinstance(value, dict) else [(*prefix, key)]
        )
    ]


def lookup(table: dict, path: tuple):
    for key in path:
        table = table[key]
    return table


@pytest.fixture
def written(modules, model, tmp_path, monkeypatch) -> dict:
    """What the serve scheduler writes as an engine with the box's CPU tier starts (its connector is the engine's
    to make: the scheduler here only reads its configuration)."""
    serve = modules[1]
    path = tmp_path / "engine-effective.json"
    monkeypatch.setenv("FLATSHARE_ENGINE_EFFECTIVE", str(path))
    sched = scheduler(serve.AsyncScheduler, model, num_blocks=64)
    first = json.loads(path.read_text())
    assert first["kv_transfer"] is None  # the suite's schedulers have no connector
    sched.vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=dict(BOX["engine"]["kv_transfer"]["kv_connector_extra_config"]),
    )
    serve.write(str(path), serve.effective(sched))
    return json.loads(path.read_text())


def test_it_writes_the_configuration_it_was_built_with(modules, written, tmp_path):
    make_room = modules[0]
    import vllm

    assert written["schema"] == 1 and written["pid"] == os.getpid()
    assert written["vllm_version"] == vllm.__version__
    assert written["scheduler_cls"] == "flatshare_make_room_serve.AsyncScheduler"
    assert written["scheduler_module"] == {
        "name": "flatshare_make_room_scheduler",
        "sha256": sha256(make_room.__file__),
    }
    assert written["serve_module_sha256"] == sha256(SERVE) == BOX["engine"]["serve_module_sha256"]
    assert (
        written["serve_modules_sha256"]
        == BOX["engine"]["serve_modules_sha256"]
        == {f"{helper}.py": sha256(SERVE.parent / f"{helper}.py") for helper in HELPERS}
    )
    # The suite's scheduler asks for neither: vLLM's queue, vLLM's copy.
    assert written["queue"] == {
        "max_wait_s": 0,
        "waiting_queue": "vllm.v1.core.sched.request_queue.PriorityRequestQueue",
    }
    assert written["copy"] == {"sm_load_programs": 0}
    assert written["make_room"] == SUITE_SWITCHES | {"reserve_blocks": 0, "retain_blocks": 8}
    assert written["jit"] == JIT
    sched = written["scheduler"]
    assert (
        sched["policy"],
        sched["max_num_batched_tokens"],
        sched["max_num_scheduled_tokens"],
    ) == (
        "priority",
        128,
        128,
    )
    assert (sched["long_prefill_token_threshold"], sched["max_num_seqs"]) == (64, 16)
    cache = written["cache"]
    assert (cache["block_size"], cache["num_gpu_blocks"], cache["pool_tokens"]) == (16, 64, 1024)
    assert written["kv_transfer"] == BOX["engine"]["kv_transfer"]
    # Every setting the box's profile names is one the scheduler writes.
    missing = []
    for path in leaf_paths(BOX["engine"]):
        try:
            lookup(written, path)
        except (KeyError, TypeError):
            missing.append(path)
    assert missing == []
    assert [path.name for path in tmp_path.iterdir()] == [
        "engine-effective.json"
    ]  # no staged file left


def check(
    serving: dict, effective: Path, tmp_path: Path, *args: str
) -> subprocess.CompletedProcess:
    profile = tmp_path / "serving.json"
    profile.write_text(json.dumps(serving))
    return subprocess.run(
        [sys.executable, str(CHECK), str(profile), "--engine", str(effective), *args],
        capture_output=True,
        text=True,
    )


def test_serving_check_passes_the_schedulers_configuration_and_fails_one_that_differs(
    written, tmp_path
):
    # A profile of the box's shape, each setting the scheduler's own value: the check passes, the engine's pid named.
    own = copy.deepcopy(BOX)
    for path in leaf_paths(BOX["engine"]):
        table = own["engine"]
        for key in path[:-1]:
            table = table[key]
        table[path[-1]] = lookup(written, path)
    effective = tmp_path / "engine-effective.json"
    effective.write_text(json.dumps(written))
    done = check(own, effective, tmp_path, "--pid", str(os.getpid()))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "serving-check: ok: engine and patches as rtx-5060-ti-serve names them" in done.stdout
    # The negative control: a profile with arm 9's 16-token window and its 2,048-token reserve fails, naming both.
    arm9 = copy.deepcopy(own)
    arm9["engine"]["make_room"] |= {"budget_cap_tokens": 16, "reserve_tokens": 2048}
    done = check(arm9, effective, tmp_path)
    assert done.returncode == 1
    assert re.search(
        r"\n  engine\.make_room\.budget_cap_tokens +16  DIFFERS: runs 48\n", done.stdout
    )
    assert done.stderr.endswith(
        "2 setting(s) differ from rtx-5060-ti-serve's: engine.make_room.budget_cap_tokens, "
        "engine.make_room.reserve_tokens\n"
    )
    # A file another process wrote, as an earlier start leaves one, is no engine's account of itself.
    done = check(own, effective, tmp_path, "--pid", str(os.getpid() + 1_000_000))
    assert done.returncode == 2 and "a file of an earlier start" in done.stderr


def test_it_installs_the_profiles_just_in_time_scheduling(modules, model, installed):
    """With the profile's flatshare_jit, the constructor installs it on vLLM's engine core with the margin in seconds
    and the step-time model; with a margin of 0, the negative control, it installs nothing and records it off."""
    from vllm.v1.engine import core

    serve = modules[1]
    scheduler(serve.AsyncScheduler, model)
    cost = serve.flatshare_jit.StepCost(**JIT["step_cost"])
    assert installed == [(core.EngineCore, JIT["margin_ms"] / 1e3, cost)]
    off = scheduler(
        serve.AsyncScheduler, model, jit={"margin_ms": 0, "step_cost": JIT["step_cost"]}
    )
    assert len(installed) == 1
    assert serve.effective(off)["jit"] == {"margin_ms": 0, "step_cost": None}
