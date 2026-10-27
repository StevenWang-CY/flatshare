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
"""The switches it serves: 2 preemptions a step, the budget cap with a 48-token window, no reserve."""


@pytest.fixture(scope="session")
def modules(tmp_path_factory):
    """make-room's module, as the variant builds it, under the name the serve module imports, and the serve module."""
    build = TR.load(TR.PACKAGE / "build.py", "flatshare_serve_build")
    path = build.build("make-room", tmp_path_factory.mktemp("make-room"), source_tree=True)
    make_room = TR.load(path, "flatshare_make_room_scheduler")
    return make_room, TR.load(SERVE, "flatshare_make_room_serve")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scheduler(cls, model, num_blocks=48):
    return T.scheduler(
        cls,
        model,
        num_blocks,
        dict(SWITCHES),
        max_num_batched_tokens=128,
        long_prefill_token_threshold=64,
        prefix_caching=True,
    )


@pytest.mark.parametrize("num_blocks", [48, 64])
@pytest.mark.parametrize("workload", ["suite", "pressure"])
def test_it_schedules_as_make_rooms_async_scheduler(modules, model, workload, num_blocks):
    make_room, serve = modules
    # The class defines its constructor and nothing else: every other method is make-room's.
    assert serve.AsyncScheduler.__mro__[1] is make_room.AsyncScheduler
    defined = [
        name
        for name, value in vars(serve.AsyncScheduler).items()
        if callable(value) or isinstance(value, staticmethod | classmethod | property)
    ]
    assert defined == ["__init__"]
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
    assert written["make_room"] == SWITCHES | {"reserve_blocks": 0}
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
