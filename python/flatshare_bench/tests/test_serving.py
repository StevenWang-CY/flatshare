"""The box's serving profile (deploy/profiles/rtx-5060-ti-serve.toml): it renders the serving variant's drop-in and
serving.json, which hold every setting the engine must run with and its required revisions, the measured configuration
of the sweep's PR point at the step cap it chose (W5's but for the step cap), every setting of which a point's record
keeps but those UNRECORDED names; the drop-in runs exactly what serving.json names, its step cap over the frozen box's
engine.yaml; and deploy/bin/serving-check passes an engine that runs it and fails one that differs in any setting, the
negative control, as well as a file the engine did not write."""

from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from flatshare_bench.e2.record import engine_settings

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
BOX = DEPLOY / "phastform"
CHECK = DEPLOY / "bin" / "serving-check"
SERVING = json.loads((BOX / "serving.json").read_text())
MEASURED = (
    REPO
    / "benchmarks/results/2026-10-27-sweep/sweep-screen/sweep-disc-pr-ab48-b256-t0-s24/engine.json"
)
"""The record of the engine's configuration in the sweep's point of the PRs' configuration at the step cap it chose:
what engine_settings keeps of the /server_info it gave."""
W5 = REPO / "benchmarks/results/2026-10-24-e2-w5/w5-ext-r1/w5-disc-pr-ab48-s24/engine.json"
"""The same in a W5 point of the PRs' configuration, at the frozen box's step cap."""
STEP_CAP = ("max_num_batched_tokens", "long_prefill_token_threshold")
"""The step cap's two settings, the [serving] table's own."""
STEP_CAP_FLAGS = ("--max-num-batched-tokens", "--long-prefill-token-threshold")
"""The flags the variant passes them as."""

spec = importlib.util.spec_from_file_location("flatshare_profile_serving", DEPLOY / "profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
loader = importlib.machinery.SourceFileLoader("flatshare_serving_check", str(CHECK))
serving_check = importlib.util.module_from_spec(
    importlib.util.spec_from_loader(loader.name, loader)
)
loader.exec_module(serving_check)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exec_lines(text: str, key: str) -> list[str]:
    """The values of a unit's ``key=`` lines."""
    return [line.partition("=")[2] for line in text.splitlines() if line.startswith(f"{key}=")]


def effective(serving: dict = SERVING, pid: int | None = None) -> dict:
    """What the serve scheduler writes in an engine that runs ``serving``'s configuration: every setting it names, and
    the ones it does not (the dtype, the scheduler's own step cap, the reserve in blocks), written by ``pid``."""
    report = copy.deepcopy(serving["engine"])
    report["model"] |= {"dtype": "torch.bfloat16", "quantization": "fp8"}
    # The scheduler's own step cap is the configured one where no max_num_scheduled_tokens is set.
    step_cap = report["scheduler"]["max_num_batched_tokens"]
    report["scheduler"] |= {"max_num_scheduled_tokens": step_cap, "enable_chunked_prefill": True}
    report["cache"] |= {"cache_dtype": "auto"}
    report["make_room"] |= {"reserve_blocks": 0}
    return report | {
        "schema": 1,
        "pid": pid or os.getpid(),
        "ppid": os.getppid(),
        "written_at": 1_793_000_000.0,
        "additional_config": {"make_room": serving["engine"]["make_room"]},
    }


def run_check(tmp_path: Path, *args: str, serving: dict | None = None, report: dict | None = None):
    """serving-check with ``serving`` as the profile (the box's by default) and ``report`` as the engine's file."""
    profile_path = BOX / "serving.json"
    if serving is not None:
        profile_path = tmp_path / "serving.json"
        profile_path.write_text(json.dumps(serving))
    engine = tmp_path / "engine-effective.json"
    engine.write_text(json.dumps(effective() if report is None else report))
    return subprocess.run(
        [sys.executable, str(CHECK), str(profile_path), "--engine", str(engine), *args],
        capture_output=True,
        text=True,
    )


def test_the_serving_profile_renders_the_boxs_serving_files(tmp_path: Path) -> None:
    serve = profile.load("rtx-5060-ti-serve")
    assert profile.check(serve, BOX) == []
    out = tmp_path / "node"
    profile.render(serve, None, out)
    for rel in profile.SERVED:
        assert (out / rel).read_bytes() == (BOX / rel).read_bytes(), rel
    # The frozen box's own files are the base profile's, whichever renders them.
    for rel in profile.RENDERED:
        assert (out / rel).read_bytes() == (BOX / rel).read_bytes(), rel
    done = subprocess.run(
        [sys.executable, str(DEPLOY / "profile.py"), "effective", "rtx-5060-ti-serve"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == SERVING
    # A profile without a [serving] table renders none of them and has no effective configuration.
    profile.render(profile.load("rtx-5060-ti"), None, tmp_path / "frozen")
    assert not any((tmp_path / "frozen" / rel).exists() for rel in profile.SERVED)
    done = subprocess.run(
        [sys.executable, str(DEPLOY / "profile.py"), "effective", "rtx-5060-ti"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2 and "no serving profile" in done.stderr


def test_serving_json_holds_every_effective_setting_and_its_required_revisions() -> None:
    engine = SERVING["engine"]
    assert (SERVING["profile"], SERVING["variant"]) == ("rtx-5060-ti-serve", "make-room-serve")
    assert engine["vllm_version"] == "0.32.0"
    assert engine["model"] == {
        "model": "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "revision": "8591804019c8b22094c3b5b4454e0edc05dffc98",
        "tokenizer": "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "tokenizer_revision": "8591804019c8b22094c3b5b4454e0edc05dffc98",
        "max_model_len": 32768,
    }
    assert engine["scheduler"] == {
        "policy": "priority",
        "max_num_batched_tokens": 256,
        "long_prefill_token_threshold": 0,
        "max_num_seqs": 256,
        "async_scheduling": True,
    }
    assert engine["cache"] == {
        "block_size": 16,
        "num_gpu_blocks": 2275,
        "pool_tokens": 36400,
        "kv_cache_memory_bytes": 5 << 30,
        "gpu_memory_utilization": 0.7,
        "enable_prefix_caching": True,
    }
    assert engine["kv_transfer"]["kv_connector_extra_config"] == {
        "cpu_bytes_to_use": 20 << 30,
        "offload_prompt_only": False,
    }
    assert engine["make_room"] == {
        "max_preemptions_per_step": 2,
        "budget_cap": True,
        "budget_cap_tokens": 48,
        "reserve_tokens": 0,
    }
    # The scheduler: the serve module, over the module make-room's patches build, each by its hash.
    assert engine["scheduler_cls"] == "flatshare_make_room_serve.AsyncScheduler"
    assert engine["serve_module_sha256"] == sha256(profile.SERVE_MODULE)
    patches = REPO / "python/flatshare_vllm/src/flatshare_vllm/make-room"
    assert SERVING["patches"] == {
        name: sha256(patches / name) for name in ("63350.patch", "63351.patch", "variant.patch")
    }
    assert re.fullmatch(r"[0-9a-f]{64}", engine["scheduler_module"]["sha256"])
    assert SERVING["voice"] == {"mode": "serve", "max_calls": 2, "audio": "microphone"}
    # The card's envelope beside the calls in the rounds (benchmarks/results/2026-10-28-rounds): four session trees.
    assert SERVING["gateway"] == {"max_agents": 4}


COMPARED = {
    "model": ("model", "revision", "max_model_len"),
    "cache": ("block_size", "num_gpu_blocks", "kv_cache_memory_bytes", "enable_prefix_caching"),
    "scheduler": (
        "policy",
        "max_num_batched_tokens",
        "long_prefill_token_threshold",
        "max_num_seqs",
        "async_scheduling",
    ),
}
"""The settings of serving.json's engine that a measured point's engine.json records, beside its CPU tier and
make-room's switches."""


def profile_settings(engine: dict = SERVING["engine"]) -> dict:
    """serving.json's engine settings that a point's engine.json records, by their dotted names."""
    return {
        **{
            f"{table}.{key}": engine[table][key] for table, keys in COMPARED.items() for key in keys
        },
        "kv_transfer.kv_connector": engine["kv_transfer"]["kv_connector"],
        "kv_transfer.kv_connector_extra_config": engine["kv_transfer"]["kv_connector_extra_config"],
        "make_room": engine["make_room"],
    }


def served(measured: dict) -> dict:
    """A point's engine.json by serving.json's names: its model, scheduler, scheduler class, cache and CPU tier, and
    make-room's switches, which vLLM keeps in its additional configuration."""
    return {
        "scheduler_cls": measured["scheduler_config"]["scheduler_cls"],
        "model": measured["model_config"],
        "scheduler": measured["scheduler_config"],
        "cache": measured["cache_config"],
        "kv_transfer": measured["kv_transfer_config"],
        "make_room": measured["additional_config"]["make_room"],
    }


def measured_settings(measured: dict) -> dict:
    """The same settings from a point's engine.json, its model read as serving-check reads a snapshot of it in
    Hugging Face's cache, as an engine that resolved the model offline reports it."""
    model = serving_check.as_named({"model": measured["model_config"]})["model"]
    return profile_settings(served(measured) | {"model": model})


def test_the_serving_profile_is_the_configuration_the_sweep_chose() -> None:
    # The PRs' configuration at the step cap the sweep chose, as its screen ran it, from the engine's own /server_info:
    # the same model, revision, step cap, pool, CPU tier, policy, request slots and switches. Only the scheduler's
    # module differs, then make-room's, now the serve module over it.
    measured = json.loads(MEASURED.read_text())
    engine = SERVING["engine"]
    assert measured_settings(measured) == profile_settings()
    assert (
        measured["scheduler_config"]["async_scheduling"] is engine["scheduler"]["async_scheduling"]
    )
    named = serving_check.as_named({"model": measured["model_config"]})["model"]
    assert (named["tokenizer"], named["tokenizer_revision"]) == (
        engine["model"]["tokenizer"],
        engine["model"]["tokenizer_revision"],
    )
    # W5's point of the same configuration differs from it in the step cap alone, which was the frozen box's.
    w5 = measured_settings(json.loads(W5.read_text()))
    assert sorted(key for key, value in profile_settings().items() if w5[key] != value) == [
        "scheduler.long_prefill_token_threshold",
        "scheduler.max_num_batched_tokens",
    ]
    frozen = profile.load("rtx-5060-ti")["engine"]
    assert (
        w5["scheduler.max_num_batched_tokens"],
        w5["scheduler.long_prefill_token_threshold"],
    ) == (
        frozen["max_num_batched_tokens"],
        frozen["long_prefill_token_threshold"],
    )


UNRECORDED = {
    "engine.vllm_version": "the release, which a record's run.json names (its engine's version)",
    "engine.scheduler_module.name": "the build, which a record names by the revision the GPU node ran (run.json's "
    "versions), from which build.py writes the module",
    "engine.scheduler_module.sha256": "the same",
    "engine.serve_module_sha256": "the same",
    "engine.cache.pool_tokens": "the blocks times their size, as run.json's engine gives it",
    "engine.cache.gpu_memory_utilization": "where kv_cache_memory_bytes fixes the pool, as every profile does, vLLM "
    "sizes nothing by it and only checks that this share of the card is free at start",
    "engine.kv_transfer.kv_role": "deploy/templates/engine.yaml's kv_both for every profile, which no variant or arm "
    "changes",
}
"""serving.json's engine settings that a point's record (engine_settings, what flatshare-bench e2 export writes of its
engine.json) does not keep, and why. It keeps every other one (a test holds it), so that a setting the profile comes
to pin and the reduction drops fails there, until the reduction keeps it or it is here, and not later, in the
comparison with a record exported without it."""


def test_a_record_keeps_each_engine_setting_serving_json_pins_but_the_unrecorded() -> None:
    # An engine that runs serving.json reports its settings in vLLM's configuration, and a point's record keeps what
    # engine_settings reduces that to: read back by serving.json's names, as serving-check compares them, it has every
    # setting but those UNRECORDED names. 15e96e3's export dropped the tokenizer, which serving.json pins, and only the
    # test above found it, on a record that had lost it.
    engine = SERVING["engine"]
    configured = {
        "model_config": engine["model"],
        "scheduler_config": engine["scheduler"] | {"scheduler_cls": engine["scheduler_cls"]},
        "cache_config": engine["cache"],
        "kv_transfer_config": engine["kv_transfer"],
        "additional_config": {"make_room": engine["make_room"]},
    }
    kept = served(engine_settings(configured))
    dropped = [
        name
        for name, want, have in serving_check.leaves(engine, kept, "engine")
        if not serving_check.same(want, have)
    ]
    assert set(dropped) == set(UNRECORDED)


def test_the_serving_variant_runs_what_serving_json_names() -> None:
    conf = (BOX / "variants" / "make-room-serve" / "flatshare-engine.conf").read_text()
    [start] = [line for line in exec_lines(conf, "ExecStart") if line]
    words = start.split()
    assert words[:4] == [
        "%h/flatshare/src/deploy/bin/in-private-net",
        "%h/flatshare/venvs/engine/bin/vllm",
        "serve",
        "--config",
    ]
    assert words[4] == "%h/flatshare/src/deploy/phastform/engine.yaml"
    flags = dict(zip(words[5::2], words[6::2], strict=False))
    assert flags["--scheduler-cls"] == SERVING["engine"]["scheduler_cls"]
    # The step cap, once each, on the command line, which vLLM takes over engine.yaml's (the frozen box's).
    sched = SERVING["engine"]["scheduler"]
    assert [words.count(flag) for flag in STEP_CAP_FLAGS] == [1, 1]
    assert [flags[flag] for flag in STEP_CAP_FLAGS] == [
        str(sched["max_num_batched_tokens"]),
        str(sched["long_prefill_token_threshold"]),
    ]
    switches = {
        key.removeprefix("--additional-config.make_room."): value
        for key, value in flags.items()
        if key.startswith("--additional-config.make_room.")
    }
    assert switches == {
        key: str(value).lower() for key, value in SERVING["engine"]["make_room"].items()
    }
    assert words[-3:] == ["--uds", "${FLATSHARE_ENGINE_SOCKET}", "$FLATSHARE_ENGINE_FLAGS"]
    # make-room's module, built into the directory on PYTHONPATH, and the serve module's directory after it.
    environment = dict(line.split("=", 1) for line in exec_lines(conf, "Environment"))
    build, remove = exec_lines(conf, "ExecStartPre")
    assert build.endswith("/build.py make-room %h/flatshare/cache/make-room")
    serve_dir = profile.SERVE_MODULE.parent.relative_to(REPO)
    assert environment["PYTHONPATH"] == f"%h/flatshare/cache/make-room:%h/flatshare/src/{serve_dir}"
    assert environment["FLATSHARE_ENGINE_EFFECTIVE"] == "%t/flatshare/engine-effective.json"
    assert remove == "/usr/bin/rm -f ${FLATSHARE_ENGINE_EFFECTIVE}"
    # After the unit's own wait for /health: the check of what the engine wrote against the profile beside it.
    assert exec_lines(conf, "ExecStartPost") == [
        "/usr/bin/python3 %h/flatshare/src/deploy/bin/serving-check "
        "%h/flatshare/src/deploy/phastform/serving.json --engine ${FLATSHARE_ENGINE_EFFECTIVE} --pid ${MAINPID}"
    ]


def test_the_step_cap_is_the_serving_tables_own(tmp_path: Path) -> None:
    # serving.json's scheduler and the variant's command line take the [serving] table's step cap, and the node's
    # engine.yaml keeps the base profile's, the frozen box's: the box's two differ, and another step cap (its
    # threshold at the cap, the largest taken) moves the first two and leaves engine.yaml as it was.
    frozen = profile.load("rtx-5060-ti")["engine"]
    serve = profile.load("rtx-5060-ti-serve")
    own = {key: serve["serving"][key] for key in STEP_CAP}
    assert own != {key: frozen[key] for key in STEP_CAP}
    engine_yaml = (BOX / "engine.yaml").read_text()
    assert (
        f"max-num-batched-tokens: {frozen['max_num_batched_tokens']}\n"
        f"long-prefill-token-threshold: {frozen['long_prefill_token_threshold']}\n"
    ) in engine_yaml
    for name, change in (("box", {}), ("other", dict.fromkeys(STEP_CAP, 1024))):
        serve["serving"] |= change
        out = tmp_path / name
        profile.render(serve, None, out)
        want = own | change
        scheduler = json.loads((out / "serving.json").read_text())["engine"]["scheduler"]
        assert {key: scheduler[key] for key in STEP_CAP} == want
        conf = (out / "variants" / "make-room-serve" / "flatshare-engine.conf").read_text()
        [start] = [line for line in exec_lines(conf, "ExecStart") if line]
        flags = dict(re.findall(r"(--[a-z-]+) ([^-$ ]\S*)", start))
        assert [flags[flag] for flag in STEP_CAP_FLAGS] == [str(want[key]) for key in STEP_CAP]
        assert (out / "engine.yaml").read_text() == engine_yaml


def test_serving_check_fails_an_engine_that_runs_engine_yamls_step_cap(tmp_path: Path) -> None:
    # The negative control of the step cap: an engine started without the variant's two flags runs engine.yaml's, the
    # frozen box's, and the check names both settings.
    frozen = profile.load("rtx-5060-ti")["engine"]
    report = effective()
    report["scheduler"] |= {key: frozen[key] for key in STEP_CAP}
    report["scheduler"]["max_num_scheduled_tokens"] = frozen["max_num_batched_tokens"]
    done = run_check(tmp_path, report=report)
    assert done.returncode == 1, done.stdout + done.stderr
    for key in STEP_CAP:
        assert re.search(
            rf"^  engine\.scheduler\.{key} +\S+  DIFFERS: runs {frozen[key]}$", done.stdout, re.M
        ), key
    assert done.stderr == (
        "serving-check: 2 setting(s) differ from rtx-5060-ti-serve's: engine.scheduler.max_num_batched_tokens, "
        "engine.scheduler.long_prefill_token_threshold\n"
    )


def test_make_room_keeps_arm_nines_values() -> None:
    # E2's arms and records name make-room: its unit keeps arm 9's two preemptions, the 16-token window (the default,
    # no flag) and the 2,048-token reserve, and an arm's flags still change one.
    conf = (BOX / "variants" / "make-room" / "flatshare-engine.conf").read_text()
    [start] = [line for line in exec_lines(conf, "ExecStart") if line]
    assert (
        "--scheduler-cls flatshare_make_room_scheduler.AsyncScheduler "
        "--additional-config.make_room.max_preemptions_per_step 2 "
        "--additional-config.make_room.budget_cap true "
        "--additional-config.make_room.reserve_tokens 2048 "
        "--uds ${FLATSHARE_ENGINE_SOCKET} $FLATSHARE_ENGINE_FLAGS"
    ) in start
    assert "budget_cap_tokens" not in start and "serving-check" not in conf


def test_serving_check_passes_an_engine_that_runs_the_profile(tmp_path: Path) -> None:
    done = run_check(tmp_path, "--pid", str(os.getpid()))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "serving-check: ok: engine and patches as rtx-5060-ti-serve names them" in done.stdout
    # Every setting is printed, the effective configuration at a glance.
    for name in (
        "engine.make_room.budget_cap_tokens",
        "engine.cache.pool_tokens",
        "patches.variant.patch",
    ):
        assert re.search(rf"^  {re.escape(name)} +\S", done.stdout, re.M), name
    # The engine core writes it, a child of the unit's main process.
    done = run_check(tmp_path, "--pid", str(os.getppid()))
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(
    ("path", "value", "runs"),
    [
        (("make_room", "budget_cap_tokens"), 16, "16"),  # arm 9's window
        (("make_room", "reserve_tokens"), 2048, "2048"),  # and its reserve
        (("make_room", "budget_cap"), 1, "1"),  # one is not true
        (("scheduler", "max_num_batched_tokens"), 2048, "2048"),
        (("cache", "pool_tokens"), 33683, "33683"),
        (("model", "revision"), "main", '"main"'),
        (("scheduler_module", "sha256"), "0" * 64, f'"{"0" * 64}"'),
        (("vllm_version", None), "0.33.0", '"0.33.0"'),
    ],
)
def test_serving_check_fails_an_engine_that_differs_in_any_setting(
    tmp_path: Path, path: tuple, value: object, runs: str
) -> None:
    report = effective()
    table, key = (report[path[0]], path[1]) if path[1] else (report, path[0])
    table[key] = value
    done = run_check(tmp_path, report=report)
    assert done.returncode == 1, done.stdout + done.stderr
    name = "engine." + ".".join(p for p in path if p)
    assert re.search(
        rf"^  {re.escape(name)} +\S+  DIFFERS: runs {re.escape(runs)}$", done.stdout, re.M
    )
    assert done.stderr == f"serving-check: 1 setting(s) differ from rtx-5060-ti-serve's: {name}\n"


def test_a_profile_that_differs_fails_the_check_of_an_engine_that_runs_the_boxs(
    tmp_path: Path,
) -> None:
    # The negative control from the profile's side: the serving profile with make-room's arm-9 values, rendered, against
    # an engine that runs the box's profile: both differences named, and profile.py check names the two files.
    serve = profile.load("rtx-5060-ti-serve")
    serve["serving"] |= {"budget_cap_tokens": 16, "reserve_tokens": 2048}
    out = tmp_path / "node"
    profile.render(serve, None, out)
    assert profile.check(serve, BOX) == list(profile.SERVED)
    done = run_check(tmp_path, serving=json.loads((out / "serving.json").read_text()))
    assert done.returncode == 1
    assert done.stderr == (
        "serving-check: 2 setting(s) differ from rtx-5060-ti-serve's: engine.make_room.budget_cap_tokens, "
        "engine.make_room.reserve_tokens\n"
    )


def test_serving_check_holds_the_releases_patches_to_the_profiles(tmp_path: Path) -> None:
    # The module's sources: a release whose variant.patch is not the one the profile names fails, before any start
    # builds a module from it.
    root = tmp_path / "release"
    patches = root / "python/flatshare_vllm/src/flatshare_vllm/make-room"
    shutil.copytree(REPO / "python/flatshare_vllm/src/flatshare_vllm/make-room", patches)
    assert run_check(tmp_path, "--root", str(root)).returncode == 0
    (patches / "variant.patch").write_text((patches / "variant.patch").read_text() + "\n")
    done = run_check(tmp_path, "--root", str(root))
    assert done.returncode == 1
    assert done.stderr.endswith("differ from rtx-5060-ti-serve's: patches.variant.patch\n")
    (patches / "63350.patch").unlink()
    done = run_check(tmp_path, "--root", str(root))
    assert re.search(r"^  patches\.63350\.patch +\S+  DIFFERS: runs nothing$", done.stdout, re.M)


def test_serving_check_reads_a_snapshot_path_as_the_model_it_is_of(tmp_path: Path) -> None:
    # Offline, vLLM resolves the model to its snapshot in Hugging Face's cache and reports that path as the model and
    # the tokenizer, as the box's engine did at the first start of the serving variant: the snapshot of the pinned
    # revision is the pinned model, and a snapshot of another revision is not.
    report = effective()
    model = report["model"]
    revision = model["revision"]
    cache = (
        "/home/someone/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots"
    )
    resolved = model | {"model": f"{cache}/{revision}", "tokenizer": f"{cache}/{revision}"}
    done = run_check(tmp_path, report=report | {"model": resolved})
    assert done.returncode == 0, done.stdout + done.stderr
    assert re.search(
        r'^  engine\.model\.model +"Qwen/Qwen3-4B-Instruct-2507-FP8"$', done.stdout, re.M
    )
    other = "0" * 40
    stale = model | {"model": f"{cache}/{other}", "tokenizer": f"{cache}/{other}"}
    done = run_check(tmp_path, report=report | {"model": stale})
    assert done.returncode == 1
    assert re.search(
        r"^  engine\.model\.model +\S+  DIFFERS: runs .*snapshots/0{40}", done.stdout, re.M
    )


def test_serving_check_refuses_a_file_the_engine_did_not_write(tmp_path: Path) -> None:
    # A file of an earlier start (another pid), of another format, or none: exit 2, nothing compared.
    done = run_check(tmp_path, "--pid", "1")
    assert done.returncode == 2 and "a file of an earlier start" in done.stderr
    done = run_check(tmp_path, report=effective() | {"schema": 2})
    assert done.returncode == 2 and "schema 2, not 1" in done.stderr
    done = subprocess.run(
        [
            sys.executable,
            str(CHECK),
            str(BOX / "serving.json"),
            "--engine",
            str(tmp_path / "none.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2 and "no effective configuration at" in done.stderr
    (tmp_path / "broken.json").write_text("{")
    done = subprocess.run(
        [
            sys.executable,
            str(CHECK),
            str(tmp_path / "broken.json"),
            "--engine",
            str(tmp_path / "x"),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2 and "serving profile" in done.stderr


def voice_status(**fields: object) -> dict:
    """The voice worker's status file in serve mode (python/flatshare_voice's worker.py)."""
    return {
        "schema": 1,
        "pid": os.getpid(),
        "updated": 1_793_000_000.0,
        "mode": "serve",
        "max_calls": 2,
        "audio": "microphone",
        "active_calls": 0,
        "admitting": True,
    } | fields


def test_serving_check_holds_the_voice_worker_to_the_profiles_admission(tmp_path: Path) -> None:
    status = tmp_path / "voice-worker.json"

    def check(**fields: object) -> subprocess.CompletedProcess[str]:
        status.write_text(json.dumps(voice_status(**fields)))
        return subprocess.run(
            [
                *(sys.executable, str(CHECK), str(BOX / "serving.json")),
                *("--voice", str(status), "--voice-pid", str(os.getpid())),
            ],
            capture_output=True,
            text=True,
        )

    done = check()
    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        "serving-check: ok: voice worker and patches as rtx-5060-ti-serve names them" in done.stdout
    )
    done = check(max_calls=6)
    assert done.returncode == 1 and done.stderr.endswith("voice.max_calls\n")
    done = check(mode="bench")
    assert done.returncode == 1 and done.stderr.endswith("voice.mode\n")
    done = check(pid=os.getpid() + 1_000_000)
    assert done.returncode == 2 and "a file of an earlier start" in done.stderr


def gateway_nodes(calls: str = "fill", cover: int = 0, **nodes: dict) -> dict:
    """The gateway's GET /nodes answer for the serving arrangement on two cards (deploy/cyan/gateway.example.toml):
    phastform with both calls and four agents' sessions, gray a warm spare with twelve; ``nodes`` changes a node's
    entry, by name."""
    entries = {
        "phastform": {"roles": ["realtime", "background"], "max_calls": 2, "max_agents": 4},
        "gray": {"roles": ["realtime", "background"], "max_calls": 2, "max_agents": 12},
    }
    for name, change in nodes.items():
        entries[name] |= change
    return {
        "nodes": [{"name": name} | entry for name, entry in entries.items()],
        "sessions": 0,
        "placement": {"calls": calls, "failover_background_max_in_flight": cover},
    }


def test_serving_check_holds_the_gateways_nodes_to_the_profiles_envelope(tmp_path: Path) -> None:
    view = tmp_path / "gateway-nodes.json"

    def check(answer: dict) -> subprocess.CompletedProcess[str]:
        view.write_text(json.dumps(answer))
        return subprocess.run(
            [sys.executable, str(CHECK), str(BOX / "serving.json"), "--nodes", str(view)],
            capture_output=True,
            text=True,
        )

    done = check(gateway_nodes())
    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        "serving-check: ok: gateway's nodes and patches as rtx-5060-ti-serve names them"
        in done.stdout
    )
    assert re.search(r"nodes\.phastform\.max_agents +at most 4\n", done.stdout)
    # With fill the spare's twelve sessions wait while it carries calls; it is not held to the envelope, its cover is.
    assert "nodes.gray.max_agents" not in done.stdout
    for answer, setting in [
        (gateway_nodes(phastform={"max_agents": 12}), "nodes.phastform.max_agents"),
        (gateway_nodes(phastform={"max_agents": None}), "nodes.phastform.max_agents"),
        # Spread, every realtime node carries calls beside its own agents.
        (gateway_nodes("spread"), "nodes.gray.max_agents"),
        # A node that cannot take every call the worker admits loses one when the other fails.
        (gateway_nodes(gray={"max_calls": 1}), "nodes.gray.max_calls"),
        (gateway_nodes(cover=8), "placement.failover_background_max_in_flight"),
    ]:
        done = check(answer)
        assert done.returncode == 1 and done.stderr.endswith(f"{setting}\n"), done.stderr
    # A spare that takes no calls is no call node, and one node alone has no cover to hold.
    alone = gateway_nodes(gray={"roles": ["background"], "max_calls": 0})
    assert check(alone).returncode == 0
    # A gateway before the placement was in /nodes cannot be held to the envelope.
    done = check({"nodes": [], "sessions": 0})
    assert done.returncode == 2 and "no placement" in done.stderr


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"max_calls": None}, "[serving] needs max_calls, a int"),
        ({"budget_cap": "yes"}, "[serving] needs budget_cap, a bool"),
        ({"reserve_tokens": True}, "[serving] needs reserve_tokens, a int"),
        ({"window": 48}, "[serving] has no key window"),
        ({"pool_tokens": 36401}, "pool_tokens is no whole number of 16-token blocks"),
        ({"audio": "phone"}, "audio must be microphone or synthetic"),
        ({"max_calls": 0}, "max_calls must be 1 or more"),
        ({"max_agents": None}, "[serving] needs max_agents, a int"),
        ({"max_agents": 0}, "max_agents must be 1 or more"),
        ({"variant": "make-room"}, "the serving variant is make-room-serve"),
        # The step cap is the table's own: required, and refused by name out of its bounds.
        ({"max_num_batched_tokens": None}, "[serving] needs max_num_batched_tokens, a int"),
        (
            {"long_prefill_token_threshold": None},
            "[serving] needs long_prefill_token_threshold, a int",
        ),
        ({"max_num_batched_tokens": 0}, "max_num_batched_tokens must be 1 or more"),
        # vLLM 0.32.0 refuses a step cap below its request slots (256 here).
        (
            {"max_num_batched_tokens": 255, "long_prefill_token_threshold": 0},
            "max_num_batched_tokens (255) must be at least max_num_seqs (256)",
        ),
        (
            {"max_num_batched_tokens": 512, "long_prefill_token_threshold": -1},
            "long_prefill_token_threshold must be from 0 to max_num_batched_tokens (512), not -1",
        ),
        (
            {"max_num_batched_tokens": 512, "long_prefill_token_threshold": 513},
            "long_prefill_token_threshold must be from 0 to max_num_batched_tokens (512), not 513",
        ),
    ],
)
def test_a_serving_table_is_checked_as_it_renders(
    tmp_path: Path, change: dict, message: str
) -> None:
    serve = profile.load("rtx-5060-ti-serve")
    serve["serving"] |= change
    serve["serving"] = {key: value for key, value in serve["serving"].items() if value is not None}
    with pytest.raises(profile.ProfileError, match=re.escape(message)):
        profile.render(serve, None, tmp_path / "node")
    assert not (tmp_path / "node").exists()  # refused before any file is written
