"""The engine variants' build (src/flatshare_vllm/build.py) against vLLM v0.32.0's scheduler source: each variant's
patches apply as git applies them, the module compiles and is the patched scheduler and the stock AsyncScheduler, a
source the patches do not match fails the build naming the hunk, and the build refuses a vLLM other than 0.32.0.
tests/test_make_room.py checks make-room's behavior on the real scheduler, and tests/test_trace.py that it schedules
as the PRs do."""

import errno
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1] / "src/flatshare_vllm/build.py"
VLLM_SRC = os.environ.get("VLLM_SRC")
MODULES = {"make-room": "flatshare_make_room_scheduler", "pr54530": "flatshare_pr54530_scheduler"}

needs_vllm_src = pytest.mark.skipif(not VLLM_SRC, reason="needs VLLM_SRC, a vLLM v0.32.0 checkout")


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("flatshare_build", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(out: Path, *args: str, path: tuple[Path, ...] = ()) -> subprocess.CompletedProcess:
    """build.py in a clean interpreter that finds vLLM, with find_spec, in ``path`` and then VLLM_SRC."""
    env = os.environ | {"PYTHONPATH": os.pathsep.join([*map(str, path), str(VLLM_SRC)])}
    return subprocess.run(
        [sys.executable, str(BUILD), *args[:1], str(out), *args[1:]],
        env=env,
        capture_output=True,
        text=True,
    )


def installed(tmp_path: Path, version: str) -> Path:
    """A directory on which importlib.metadata finds an installed vLLM of ``version``."""
    dist_info = tmp_path / "site" / f"vllm-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n")
    return dist_info.parent


def git_applied(tmp_path: Path, patches: list[Path]) -> str:
    """v0.32.0's scheduler.py with ``patches`` applied by git, in order."""
    tree = tmp_path / "git"
    sched = tree / "vllm/v1/core/sched"
    sched.mkdir(parents=True)
    shutil.copy(Path(str(VLLM_SRC)) / "vllm/v1/core/sched/scheduler.py", sched)
    for patch in patches:
        subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)
    return (sched / "scheduler.py").read_text()


@needs_vllm_src
@pytest.mark.parametrize("variant", list(MODULES))
def test_build_applies_the_patches_and_appends_the_stock_async_scheduler(
    tmp_path, builder, variant
):
    assert build(tmp_path, variant, "--source-tree").returncode == 0
    module = (tmp_path / f"{MODULES[variant]}.py").read_text()
    compile(module, f"{MODULES[variant]}.py", "exec")
    header, body = module.split('Do not edit."""\n', 1)
    assert header.startswith('"""vLLM v1 Scheduler and AsyncScheduler with ')
    scheduler, async_scheduler = body.split("\n\n# SPDX-License-Identifier: Apache-2.0\n", 1)

    patches = [BUILD.parent / variant / name for name in builder.VARIANTS[variant][0]]
    assert scheduler == git_applied(tmp_path, patches)
    stock_async = (Path(str(VLLM_SRC)) / "vllm/v1/core/sched/async_scheduler.py").read_text()
    # Its import of the stock Scheduler, and its logger, which would rebind the module's to its own name.
    assert "# SPDX-License-Identifier: Apache-2.0\n" + async_scheduler == stock_async.replace(
        "from vllm.v1.core.sched.scheduler import Scheduler\n", ""
    ).replace("logger = init_logger(__name__)\n", "")
    assert module.count("\nclass Scheduler(") == 1
    assert module.count("\nclass AsyncScheduler(Scheduler):") == 1  # over the patched Scheduler
    loggers = [line for line in module.splitlines() if line.startswith("logger = ")]
    assert loggers[-1] == 'logger = init_logger("vllm.v1.core.sched.scheduler")'


@needs_vllm_src
def test_make_room_is_the_prs_and_the_variants_own_parts(tmp_path):
    assert build(tmp_path, "make-room", "--source-tree").returncode == 0
    module = (tmp_path / "flatshare_make_room_scheduler.py").read_text()
    for line in (
        # #63350's and #63351's.
        "make_room_req = self._peek_make_room_request(defer_prefills)",
        "make_room_preempted_reqs = self._make_room(",
        "first_tokens_priority = self._first_tokens_priority(",
        "self._make_room_untried.discard(request)",
        "self._num_priority_preemptions += len(victims)",
        "self._count_prefill_hold(0)",
        # The variant's: the switches from --additional-config, the reserve, the counters' log.
        "make_room = self._make_room_switches(",
        "num_blocks += self.reserve_blocks",
        "self.make_room_held_admissions += 1",
        "self._make_room_log()",
    ):
        assert line in module
    assert "self.scheduler_config.max_priority_preemptions_per_step" not in module


@needs_vllm_src
def test_a_source_the_patches_do_not_match_fails_the_build_naming_the_hunk(tmp_path):
    """A moved line fails loudly: here 0.32.0's scheduler with one context line of #63350's first hunk edited."""
    fake = tmp_path / "src/vllm"
    shutil.copytree(Path(str(VLLM_SRC)) / "vllm/v1/core/sched", fake / "v1/core/sched")
    (fake / "__init__.py").write_text("")
    scheduler = fake / "v1/core/sched/scheduler.py"
    source = scheduler.read_text()
    line = "            self.scheduler_config.scheduler_reserve_full_isl\n"
    assert source.count(line) == 1
    scheduler.write_text(
        source.replace(
            line, "            self.scheduler_config.scheduler_reserve_full_isl  # moved\n"
        )
    )
    result = subprocess.run(
        [sys.executable, str(BUILD), "make-room", str(tmp_path / "out"), "--source-tree"],
        env=os.environ | {"PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "make-room: 63350.patch: hunk @@ -371,6 +377,34 @@" in result.stderr
    assert "does not match the scheduler at its line 371" in result.stderr
    assert not (tmp_path / "out").exists()


@needs_vllm_src
def test_build_refuses_another_installed_vllm(tmp_path):
    """The patches match 0.32.0's source: another installed version fails the build, even if every hunk still
    applies, as they do in VLLM_SRC here; 0.32.0 builds."""
    result = build(tmp_path / "out", "make-room", path=(installed(tmp_path / "other", "0.31.0"),))
    assert result.returncode != 0
    assert "written against vLLM 0.32.0, and vLLM 0.31.0 is installed" in result.stderr
    assert not (tmp_path / "out").exists()
    for variant in MODULES:
        result = build(tmp_path / "ok", variant, path=(installed(tmp_path / variant, "0.32.0"),))
        assert result.returncode == 0, result.stderr


@needs_vllm_src
def test_build_needs_source_tree_for_a_checkout(tmp_path):
    """A source checkout has no package metadata to check: only --source-tree builds from one."""
    result = build(tmp_path / "out", "pr54530")
    assert result.returncode != 0 and "--source-tree builds from a source checkout" in result.stderr
    assert build(tmp_path / "out", "pr54530", "--source-tree").returncode == 0


@needs_vllm_src
def test_a_build_cut_short_leaves_the_previous_module_whole(tmp_path, builder, monkeypatch):
    """The engine imports the module at every start: a build that fails as it writes (a full disk, say) leaves the
    previous module as it was and nothing of its own, and the next build replaces the module whole."""
    sched = Path(str(VLLM_SRC)) / "vllm/v1/core/sched"
    monkeypatch.setattr(builder, "sched_dir", lambda source_tree: sched)
    target = tmp_path / f"{MODULES['make-room']}.py"
    target.write_text("# the previous build\n")
    write = Path.write_text

    def cut_short(path, text, *args, **kwargs):
        write(path, text[: len(text) // 2])
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(Path, "write_text", cut_short)
    with pytest.raises(OSError):
        builder.build("make-room", tmp_path, source_tree=True)
    assert target.read_text() == "# the previous build\n"
    assert [path.name for path in tmp_path.iterdir()] == [target.name]
    monkeypatch.setattr(Path, "write_text", write)
    assert builder.build("make-room", tmp_path, source_tree=True) == target
    assert [path.name for path in tmp_path.iterdir()] == [target.name]
    compile(target.read_text(), target.name, "exec")


def test_build_names_its_variants(tmp_path):
    result = subprocess.run(
        [sys.executable, str(BUILD), "make_room", str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "usage: build.py {make-room,pr54530} OUT_DIR [--source-tree]" in result.stderr


DIFF = "diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py\n"


def patch(*hunks: str, head: str = DIFF) -> str:
    return (
        "What it is.\n\n"
        + head
        + "--- a/vllm/v1/core/sched/scheduler.py\n+++ b/vllm/v1/core/sched/scheduler.py\n"
        + "".join(hunks)
    )


def test_hunks_apply_at_their_lines_with_the_lines_before_them_moved(builder):
    source = "".join(f"line {n}\n" for n in range(1, 11))
    applied = builder.apply_patch(
        source,
        patch(
            "@@ -2,2 +2,3 @@\n line 2\n+new a\n line 3\n",
            "@@ -8,3 +9,2 @@ def f():\n line 8\n-line 9\n line 10\n",
        ),
        "p",
    )
    assert applied == "".join(
        f"line {n}\n" + ("new a\n" if n == 2 else "") for n in range(1, 11) if n != 9
    )


@pytest.mark.parametrize(
    "bad, message",
    [
        (
            patch("@@ -3,2 +3,3 @@\n line 2\n+new\n line 3\n"),
            "p: hunk @@ -3,2 +3,3 @@ does not match",
        ),
        (
            patch("@@ -2,2 +2,2 @@\n line 2\n-line 4\n+new\n"),
            "does not match the scheduler at its line 2",
        ),
        (patch("@@ -2,2 +2,3 @@\n line 2\n+new\n"), "p: hunk @@ -2,2 +2,3 @@ ends early"),
        (
            patch("@@ -2,1 +2,1 @@\n-line 2\n\\ No newline at end of file\n+new\n"),
            "ends with a newline",
        ),
        (
            patch("@@ -2,1 +2,1 @@\n-line 2\n+new\n", head="diff --git a/x.py b/x.py\n"),
            "one diff, of",
        ),
        (patch("@@ -2,1 +2,1 @@\n-line 2\n+new\n") + DIFF, "one diff, of"),
        (
            patch("@@ -2,1 +2,1 @@\n-line 2\n+new\n", "trailing text\n"),
            "expected a hunk at its line 9",
        ),
    ],
)
def test_a_patch_that_does_not_apply_exactly_fails(builder, bad, message):
    source = "".join(f"line {n}\n" for n in range(1, 11))
    with pytest.raises(builder.PatchError, match=message.replace("+", r"\+").replace(".", r"\.")):
        builder.apply_patch(source, bad, "p")
