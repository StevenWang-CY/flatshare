"""scripts/compat_upstream.py (make compat-upstream) against a local stand-in for vLLM's repository: a bare repository
with a main branch, the PRs' heads as refs/pull/N/head and release tags, served blobless over file://. Clean merges
and conflicts, the PRs' tests on a merge with a stand-in interpreter (the import check answered, pytest real), a
failure main has too, an interpreter that cannot import main's scheduler, no network, an older git; the backport
against the PRs' heads, whose lines in the scheduler are the backport's but for its adaptations, as upstream; and
make-room's patches on releases (with VLLM_SRC, v0.32.0's real scheduler), and each hunk's fit by git apply. It needs
git 2.38 or newer (GIT names one)."""

import difflib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compat_upstream.py"
spec = importlib.util.spec_from_file_location("flatshare_compat_upstream", SCRIPT)
compat_upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat_upstream)
GIT = os.environ.get("GIT", "git")
# git as the tests run it, the fake upstream's and the script's alike: without the developer's own config (commit
# signing, hooks), which the tests do not test.
ISOLATED = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
VLLM_SRC = os.environ.get("VLLM_SRC")
SCHED = "vllm/v1/core/sched"


def git_version() -> tuple[int, ...]:
    try:
        out = subprocess.run([GIT, "version"], capture_output=True, text=True).stdout
    except OSError:
        return (0, 0)
    match = re.search(r"(\d+)\.(\d+)", out)
    return tuple(map(int, match.groups())) if match else (0, 0)


# The tests that run the script against a fake upstream need git merge-tree --write-tree; the rest run on any git.
needs_merge_tree = pytest.mark.skipif(
    git_version() < (2, 38),
    reason="needs git 2.38 or newer (merge-tree --write-tree); GIT names one",
)
needs_vllm_src = pytest.mark.skipif(not VLLM_SRC, reason="needs VLLM_SRC, a vLLM v0.32.0 checkout")


@pytest.fixture(scope="module")
def compat():
    return compat_upstream


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [GIT, "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ | ISOLATED,
    ).stdout.strip()


class Upstream:
    """A work repository whose commits become a bare repository served as vLLM's."""

    def __init__(self, root: Path):
        self.work, self.bare = root / "work", root / "upstream.git"
        self.work.mkdir(parents=True)
        git(self.work, "init", "-q", "-b", "main")
        git(self.work, "config", "user.name", "test")
        git(self.work, "config", "user.email", "test@localhost")
        self.refs: dict[str, str] = {}

    def commit(self, parent: str | None, files: dict[str, str], message: str = "change") -> str:
        """A commit of ``files`` (None deletes one) on ``parent``."""
        if parent:
            git(self.work, "checkout", "-q", "--detach", parent)
        for name, text in files.items():
            path = self.work / name
            if text is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        git(self.work, "add", "-A")
        git(self.work, "commit", "-q", "-m", message)
        return git(self.work, "rev-parse", "HEAD")

    def serve(self, main: str, prs: dict[int, str], tags: dict[str, str]) -> str:
        """The bare repository with ``main``, the PRs' heads and ``tags``; its file:// URL."""
        if self.bare.exists():
            shutil.rmtree(self.bare)
        git(self.work, "branch", "-f", "main", main)
        git(self.work.parent, "clone", "-q", "--bare", str(self.work), str(self.bare))
        git(self.bare, "config", "uploadpack.allowFilter", "true")
        git(self.bare, "config", "uploadpack.allowAnySHA1InWant", "true")
        for pr, head in prs.items():
            git(self.bare, "update-ref", f"refs/pull/{pr}/head", head)
        for tag, rev in tags.items():
            git(self.bare, "tag", "-f", tag, rev)
        return self.bare.as_uri()


def scheduler_files(real: bool, shift: int = 0) -> dict[str, str]:
    """vllm's scheduler and async_scheduler: v0.32.0's from VLLM_SRC, ``shift`` comment lines added at the top of
    the scheduler, or stand-ins."""
    if not real:
        return {f"{SCHED}/scheduler.py": "x = 1\n", f"{SCHED}/async_scheduler.py": "y = 1\n"}
    src = Path(str(VLLM_SRC)) / SCHED
    scheduler = (src / "scheduler.py").read_text()
    return {
        f"{SCHED}/scheduler.py": "# moved\n" * shift + scheduler,
        f"{SCHED}/async_scheduler.py": (src / "async_scheduler.py").read_text(),
    }


def pr_lines(pr: int) -> str:
    """The lines #pr adds to the scheduler, one a line, as compat_upstream compares them with the backport: its patch's,
    with the PR's own where ADAPTED names an adaptation."""
    patch = next(p for p in compat_upstream.PATCHES if p.name == f"{pr}.patch")
    lines = compat_upstream.added_lines(patch.read_text())
    for theirs, ours in compat_upstream.ADAPTED[patch.name].values():
        lines = lines - Counter(ours) + Counter(theirs)
    return "".join(f"{line}\n" for line in sorted(lines.elements()))


def heads(scheduler: str) -> tuple[dict[str, str], dict[str, str]]:
    """The scheduler of #63350's head on ``scheduler``, its base's, and of #63351's on #63350's: each with the lines its
    PR adds, so that the backport is the PRs' heads but for its adaptations, as it is upstream."""
    kv = scheduler + pr_lines(63350)
    return {f"{SCHED}/scheduler.py": kv}, {f"{SCHED}/scheduler.py": kv + pr_lines(63351)}


STAND_IN = scheduler_files(real=False)[f"{SCHED}/scheduler.py"]
KV, CAP = heads(STAND_IN)
"""The PRs' heads' scheduler, #63351's stacked on #63350's, on the stand-in one."""

BASE = {
    "vllm/__init__.py": "",
    "vllm/other.py": "a = 0\n",
    "tests/test_core.py": "def test_core():\n    assert True\n",
}


@pytest.fixture
def upstream(tmp_path):
    return Upstream(tmp_path / "upstream")


@pytest.fixture
def python(tmp_path):
    """An interpreter for the PRs' tests: the import check answered (FAKE_IMPORT_ERROR fails it), pytest real."""
    path = tmp_path / "python"
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then\n'
        '  [ -z "$FAKE_IMPORT_ERROR" ] && exit 0\n'
        '  echo "$FAKE_IMPORT_ERROR" >&2; exit 1\n'
        "fi\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    path.chmod(0o755)
    return str(path)


def run(tmp_path, remote: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    clone = tmp_path / "cache/vllm-upstream"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--clone",
            str(clone),
            "--git",
            GIT,
            "--remote",
            remote,
            *args,
        ],
        capture_output=True,
        text=True,
        env=os.environ | ISOLATED | (env or {}),
    )


def standard(upstream: Upstream, pr_tests: str = "def test_pr():\n    assert True\n") -> str:
    """main moved on from the PRs' base; #63350 adds a test file, #63351 (on it) changes another file, and each adds
    its lines to the scheduler; one release, which the stand-in scheduler does not take make-room's patches."""
    base = upstream.commit(None, BASE | scheduler_files(real=False), "base")
    main = upstream.commit(base, {"vllm/main_only.py": "m = 1\n"}, "main moves on")
    kv = upstream.commit(
        base, {"vllm/kv.py": "k = 1\n", "tests/test_pr.py": pr_tests} | KV, "#63350"
    )
    cap = upstream.commit(kv, {"vllm/cap.py": "c = 1\n"} | CAP, "#63351")
    return upstream.serve(main, {63350: kv, 63351: cap}, {"v0.40.0": base})


@needs_merge_tree
def test_clean_merges_run_the_tests_the_prs_add(tmp_path, upstream, python):
    proc = run(tmp_path, standard(upstream), "--python", python)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "#63350: merges into main cleanly" in proc.stdout
    assert "#63351: merges into main cleanly" in proc.stdout
    assert "#63350: tests/test_pr.py: 1 passed" in proc.stdout
    assert "#63351: tests/test_pr.py: 1 passed" in proc.stdout  # #63351's diff holds #63350's
    assert (
        "backport = the PRs' heads but for 15 adapted lines of theirs and 15 of its own, which the patches' "
        "headers name\n" in proc.stdout
    )
    # Its stand-in scheduler is not v0.32.0's: reported, and no failure.
    assert re.search(
        r"patches on v0\.40\.0: do not apply \(make-room: 63350\.patch: hunk", proc.stdout
    )


@needs_merge_tree
def test_without_an_interpreter_no_test_runs(tmp_path, upstream):
    proc = run(tmp_path, standard(upstream))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "the PRs' tests: not run" in proc.stdout
    assert "passed" not in proc.stdout


@needs_merge_tree
def test_a_conflict_names_its_files_and_fails(tmp_path, upstream, python):
    base = upstream.commit(None, BASE | scheduler_files(real=False), "base")
    main = upstream.commit(base, {"vllm/other.py": "a = 1\n"}, "main changes other.py")
    kv = upstream.commit(base, {"vllm/kv.py": "k = 1\n"} | KV, "#63350")
    cap = upstream.commit(
        kv, {"vllm/other.py": "a = 2\n", "vllm/cap.py": "c = 1\n"} | CAP, "#63351"
    )
    remote = upstream.serve(main, {63350: kv, 63351: cap}, {"v0.40.0": base})
    proc = run(tmp_path, remote, "--python", python)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "#63350: merges into main cleanly" in proc.stdout
    assert "#63351: CONFLICTS with main in vllm/other.py\n" in proc.stdout


@needs_merge_tree
def test_a_failing_test_fails_and_one_main_fails_too_is_mains(tmp_path, upstream, python):
    core = "def test_core():\n    assert True\n\n\ndef test_broken_on_main():\n    assert False\n"
    base = upstream.commit(None, BASE | scheduler_files(real=False) | {"tests/test_core.py": core})
    main = upstream.commit(base, {"vllm/main_only.py": "m = 1\n"})
    # #63350 changes tests/test_core.py, adding a test that passes; #63351 adds one that fails.
    kv = upstream.commit(
        base, {"tests/test_core.py": core + "\n\ndef test_kv():\n    assert True\n"} | KV
    )
    cap = upstream.commit(kv, {"tests/test_cap.py": "def test_cap():\n    assert 1 == 2\n"} | CAP)
    remote = upstream.serve(main, {63350: kv, 63351: cap}, {"v0.40.0": base})
    proc = run(tmp_path, remote, "--python", python)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    kv_line = next(line for line in lines if line.startswith("  #63350:"))
    assert "1 failed, 2 passed" in kv_line and "(1 of them fail on main too)" in kv_line
    failed = [line.strip() for line in lines if line.strip().startswith("FAILED")]
    assert failed == ["FAILED tests/test_cap.py::test_cap"], proc.stdout


@needs_merge_tree
def test_a_pr_whose_tests_all_skip_has_not_been_checked(tmp_path, upstream, python):
    """A test file that skips as a whole (a CUDA-only mark, an importorskip of what main's environment lacks) passes
    no test: that is no evidence, so the check exits 2, not 0."""
    skipped = "import pytest\n\npytestmark = pytest.mark.skip(reason='CUDA only')\n\n\ndef test_pr():\n    pass\n"
    proc = run(tmp_path, standard(upstream, pr_tests=skipped), "--python", python)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "#63350: tests/test_pr.py: 1 skipped" in proc.stdout
    assert "no test ran" in proc.stdout


@needs_merge_tree
def test_a_pr_test_that_hangs_fails_its_file_when_the_time_runs_out(tmp_path, upstream, python):
    base = upstream.commit(None, BASE | scheduler_files(real=False))
    main = upstream.commit(base, {"vllm/main_only.py": "m = 1\n"})
    hangs = "import time\n\n\ndef test_pr():\n    time.sleep(120)\n"
    kv = upstream.commit(base, {"tests/test_pr.py": hangs} | KV)
    cap = upstream.commit(
        base, {"vllm/cap.py": "c = 1\n", f"{SCHED}/scheduler.py": STAND_IN + pr_lines(63351)}
    )
    remote = upstream.serve(main, {63350: kv, 63351: cap}, {"v0.40.0": base})
    proc = run(tmp_path, remote, "--python", python, "--timeout", "4")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "#63350: tests/test_pr.py: timed out after 4 s\n" in proc.stdout
    assert "FAILED tests/test_pr.py: timed out after 4 s" in proc.stdout
    assert "#63351: no test file added or changed" in proc.stdout


@needs_merge_tree
def test_an_interpreter_that_cannot_import_mains_scheduler_runs_nothing(tmp_path, upstream, python):
    error = "ModuleNotFoundError: No module named 'torch'"
    proc = run(tmp_path, standard(upstream), "--python", python, env={"FAKE_IMPORT_ERROR": error})
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert f"cannot import vLLM main's scheduler on CPU ({error}): no test ran" in proc.stdout
    assert "passed" not in proc.stdout and "FAILED" not in proc.stdout


@needs_merge_tree
def test_an_interpreter_that_cannot_run_mains_tests_runs_nothing(tmp_path, upstream, python):
    base = upstream.commit(None, BASE | scheduler_files(real=False))
    main = upstream.commit(base, {"tests/conftest.py": "import not_installed_here\n"})
    kv = upstream.commit(base, {"tests/test_pr.py": "def test_pr():\n    assert True\n"} | KV)
    cap = upstream.commit(kv, CAP)
    remote = upstream.serve(main, {63350: kv, 63351: cap}, {"v0.40.0": base})
    proc = run(tmp_path, remote, "--python", python)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "cannot run vLLM main's tests in tests (" in proc.stdout
    assert "No module named 'not_installed_here'" in proc.stdout


@needs_merge_tree
def test_a_pr_main_squash_merged_is_reported_and_not_merged_again(tmp_path, upstream):
    """vLLM squash-merges: main holds #63350 as a commit of its own whose subject ends with (#63350), and the PR's
    head is no ancestor of main. Main edits the same file afterwards, so merging the stale head would conflict, and
    so would #63351's head, which carries #63350's commits: its own changes, on #63350's head, merge."""
    base = upstream.commit(None, BASE | scheduler_files(real=False))
    kv = upstream.commit(base, {"vllm/kv.py": "k = 1\n"} | KV, "Make room under KV pressure")
    cap = upstream.commit(kv, {"vllm/cap.py": "c = 1\n"} | CAP, "Cap prefill")
    squashed = upstream.commit(
        base, {"vllm/kv.py": "k = 1\n"} | KV, "[Core] Make room under KV pressure (#63350)"
    )
    main = upstream.commit(squashed, {"vllm/kv.py": "k = 2\n"}, "Tune it (#70001)")
    proc = run(tmp_path, upstream.serve(main, {63350: kv, 63351: cap}, {"v0.40.0": base}))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    short = git(upstream.work, "rev-parse", "--short=10", squashed)
    assert f"#63350: squash-merged into main as {short}\n" in proc.stdout
    assert (
        "#63351: its own changes merge into main cleanly, on #63350 as main holds it\n"
        in proc.stdout
    )
    assert "CONFLICTS" not in proc.stdout


@needs_merge_tree
def test_a_stacked_prs_own_conflict_with_main_is_reported(tmp_path, upstream):
    base = upstream.commit(None, BASE | scheduler_files(real=False))
    kv = upstream.commit(base, {"vllm/kv.py": "k = 1\n"} | KV, "Make room")
    cap = upstream.commit(kv, {"vllm/other.py": "a = 2\n"} | CAP, "Cap prefill")
    squashed = upstream.commit(base, {"vllm/kv.py": "k = 1\n"} | KV, "Make room (#63350)")
    main = upstream.commit(squashed, {"vllm/other.py": "a = 3\n"}, "Other (#70002)")
    proc = run(tmp_path, upstream.serve(main, {63350: kv, 63351: cap}, {"v0.40.0": base}))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    expected = (
        "#63351: its own changes CONFLICT with main in vllm/other.py, on #63350 as main holds it\n"
    )
    assert expected in proc.stdout


@needs_merge_tree
def test_a_pr_merged_into_main_or_whose_changes_main_has_is_reported(tmp_path, upstream):
    base = upstream.commit(None, BASE | scheduler_files(real=False))
    kv = upstream.commit(base, {"vllm/kv.py": "k = 1\n"})
    # #63351's change, its lines in the scheduler among it, made on main under a subject that does not name it; and a
    # true merge of #63350, kv an ancestor of main, which leaves the backport nothing to compare its head from.
    change = {"vllm/main_only.py": "m = 1\n", f"{SCHED}/scheduler.py": STAND_IN + pr_lines(63351)}
    main = upstream.commit(kv, change)
    cap = upstream.commit(base, change)
    proc = run(tmp_path, upstream.serve(main, {63350: kv, 63351: cap}, {"v0.40.0": base}))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "#63350: merged into main\n" in proc.stdout
    assert "#63351: its changes are in main already\n" in proc.stdout
    assert (
        "  #63350: main holds its head, which leaves no base to compare its changes from\n"
        in proc.stdout
    )
    assert (
        "backport = the PRs' heads but for 11 adapted lines of theirs and 14 of its own"
        in proc.stdout
    )


@needs_merge_tree
def test_offline_it_fails_saying_it_needs_the_network(tmp_path, upstream):
    nowhere = (tmp_path / "nowhere.git").as_uri()
    proc = run(tmp_path, nowhere)
    assert proc.returncode == 2
    assert f"compat-upstream: cannot fetch from {nowhere}" in proc.stderr
    assert "needs the network" in proc.stderr
    # With the clone cached, it fetches again rather than check stale heads.
    assert run(tmp_path, standard(upstream)).returncode == 0
    shutil.rmtree(upstream.bare)
    proc = run(tmp_path, upstream.bare.as_uri())
    assert proc.returncode == 2 and "needs the network" in proc.stderr
    assert "vLLM main" not in proc.stdout


def test_an_older_git_is_refused(tmp_path):
    old = tmp_path / "git"
    old.write_text("#!/bin/sh\necho 'git version 2.34.1'\n")
    old.chmod(0o755)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--clone", str(tmp_path / "clone"), "--git", str(old)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "git merge-tree --write-tree needs git 2.38 or newer" in proc.stderr
    assert "is 2.34" in proc.stderr


def test_the_upstream_builds_whatever_the_developers_git_config_says(tmp_path, monkeypatch):
    """The fake upstream's commits, and the script the tests run, ignore the developer's own git config: signing that
    cannot sign, hooks that are not there."""
    hostile = tmp_path / "gitconfig"
    hostile.write_text(
        "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n[core]\n\thooksPath = /nowhere\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile))
    upstream = Upstream(tmp_path / "upstream")
    assert len(upstream.commit(None, {"vllm/__init__.py": ""})) == 40


def test_the_newest_release_and_a_newer_candidate(compat):
    tags = ["v0.31.0", "v0.32.0", "v0.32.0rc3", "v0.32.1rc0", "v0.32.1rc1", "v0.10.1.1", "v1-beta"]
    assert compat.newest(tags) == ["v0.32.0", "v0.32.1rc1"]
    assert compat.newest(["v0.32.0", "v0.32.0rc3", "v0.31.1rc0"]) == ["v0.32.0"]
    assert compat.newest(["v0.32.1rc0", "v0.33.0", "v0.32.0"]) == ["v0.33.0"]
    assert compat.newest(["v0.10.1.1", "v0.9.2"]) == ["v0.10.1.1"]
    assert compat.newest(["v0.33.0rc0"]) == []


@needs_merge_tree
@needs_vllm_src
def test_the_patches_apply_to_the_pinned_release_and_a_shifted_one_names_its_hunk(
    tmp_path, upstream
):
    base = upstream.commit(None, BASE | scheduler_files(real=True), "v0.32.0")
    rc = upstream.commit(base, scheduler_files(real=True, shift=2), "v0.32.1rc0")
    on_kv, on_cap = heads(scheduler_files(real=True)[f"{SCHED}/scheduler.py"])
    kv = upstream.commit(base, {"vllm/kv.py": "k = 1\n"} | on_kv)
    cap = upstream.commit(kv, on_cap)
    proc = run(
        tmp_path, upstream.serve(rc, {63350: kv, 63351: cap}, {"v0.32.0": base, "v0.32.1rc0": rc})
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "make-room's patches on v0.32.0, the pinned version: they apply\n" in proc.stdout
    assert (
        "make-room's patches on v0.32.1rc0: do not apply (make-room: 63350.patch: hunk @@ -53,6 +53,12 @@"
        in proc.stdout
    )
    assert (
        "does not match the scheduler at its line 53); its scheduler.py against v0.32.0: "
        "1 file changed, 2 insertions(+)." in proc.stdout
    )
    # And every hunk as git apply finds it: each of the 24 two lines down, with 3 lines of context as with 1.
    assert (
        "  each hunk by git apply, the patches in order, with 3 lines of context and with 1: "
        "24 and 24 of 24 apply\n" in proc.stdout
    )
    assert (
        "    63350.patch #1 @@ -53,6 +53,12 @@: -C3 at 55 (offset 2 lines); -C1 at 55 (offset 2 lines)\n"
        in proc.stdout
    )
    assert proc.stdout.count(" (offset 2 lines); -C1 at ") == 24


@needs_merge_tree
@needs_vllm_src
def test_patches_that_do_not_apply_to_the_pinned_release_fail(tmp_path, upstream):
    base = upstream.commit(None, BASE | scheduler_files(real=True, shift=1), "v0.32.0, edited")
    on_kv, on_cap = heads(scheduler_files(real=True, shift=1)[f"{SCHED}/scheduler.py"])
    kv = upstream.commit(base, {"vllm/kv.py": "k = 1\n"} | on_kv)
    cap = upstream.commit(kv, on_cap)
    proc = run(tmp_path, upstream.serve(base, {63350: kv, 63351: cap}, {"v0.32.0": base}))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert (
        "make-room's patches on v0.32.0, the pinned version: FAIL: make-room: 63350.patch"
        in proc.stdout
    )
    assert "with 1: 24 and 24 of 24 apply\n" in proc.stdout


def unified(before: list[str], after: list[str]) -> str:
    """A diff of the scheduler, as git writes one, from ``before`` to ``after`` (lines without their newlines)."""
    lines = difflib.unified_diff(
        [f"{line}\n" for line in before],
        [f"{line}\n" for line in after],
        f"a/{SCHED}/scheduler.py",
        f"b/{SCHED}/scheduler.py",
    )
    return f"diff --git a/{SCHED}/scheduler.py b/{SCHED}/scheduler.py\n" + "".join(lines)


def test_each_hunks_fit_is_where_git_apply_finds_it(tmp_path, compat):
    """A newer release takes the patches hunk by hunk as git apply finds them, in order, with all of a hunk's context
    and with one line of it: a hunk where it was, one that moved, one whose outer context changed and so needs less, one
    whose removed line changed and fits nowhere, and the next patch's hunk, which rests on one rejected before it."""
    pinned = [f"line {n}" for n in range(1, 41)]
    first = [*pinned[:5], "one", *pinned[5:20], "two", *pinned[20:34], *pinned[35:]]
    second = [*first[:22], "three", *first[22:]]  # after "two"
    patches = [tmp_path / "first.patch", tmp_path / "second.patch"]
    patches[0].write_text(unified(pinned, first))
    patches[1].write_text(unified(first, second))
    # The release: two lines added after line 10, and lines 18 and 35 edited.
    release = [*pinned[:10], "new", "new", *pinned[10:]]
    release[19], release[36] = "line 18, edited", "line 35, edited"
    source = "".join(f"{line}\n" for line in release)
    assert compat.fits(GIT, source, 3, patches) == [
        ("first.patch #1 @@ -3,6 +3,7 @@", "at 3"),
        ("first.patch #2 @@ -18,6 +19,7 @@", "rejected"),
        ("first.patch #3 @@ -32,7 +34,6 @@", "rejected"),
        ("second.patch #1 @@ -20,6 +20,7 @@", "rejected"),
    ]
    # The second hunk is two lines down, which git gives as 4: for each line of context it leaves out, it looks for the
    # hunk a line further up, and the hunk starts a line further down.
    assert compat.fits(GIT, source, 1, patches) == [
        ("first.patch #1 @@ -3,6 +3,7 @@", "at 3"),
        ("first.patch #2 @@ -18,6 +19,7 @@", "at 22 (offset 4 lines), context reduced to 2/2"),
        ("first.patch #3 @@ -32,7 +34,6 @@", "rejected"),
        ("second.patch #1 @@ -20,6 +20,7 @@", "at 22 (offset 2 lines)"),
    ]
    # A report that does not account for every hunk is no fit: here git found no file to patch.
    other = tmp_path / "other.patch"
    other.write_text(patches[0].read_text().replace(f"{SCHED}/scheduler.py", "vllm/other.py"))
    with pytest.raises(compat.CannotCheck, match=r"git apply -C3 other\.patch: .*vllm/other\.py"):
        compat.fits(GIT, source, 3, [other])


def test_the_lines_a_diff_adds_are_net_of_those_it_removes(compat):
    # A patch's header names no line; a line moved is in neither count; blank lines aside, each line stripped, and a
    # removed line that starts with two dashes is no file header.
    diff = (
        "vLLM #1, backported to v0.32.0: a header.\n\n"
        "diff --git a/s.py b/s.py\n--- a/s.py\n+++ b/s.py\n"
        "@@ -1,3 +1,5 @@\n"
        " def f():\n"
        "-    moved = 1\n"
        "+    added = 2\n"
        "+\n"
        "     kept = 3\n"
        "+    moved = 1\n"
        "@@ -9,1 +11,1 @@\n"
        "--- a comment\n"
        "+-- a comment, changed\n"
    )
    assert compat.added_lines(diff) == Counter({"added = 2": 1, "-- a comment, changed": 1})


def test_a_pr_and_its_patch_differ_only_where_adapted_names_them(compat):
    adapted = {"0123abcde": (("stats(n)",), ("count(n)",))}
    theirs = Counter({"a": 1, "b": 1, "stats(n)": 1})
    ours = Counter({"a": 1, "b": 1, "count(n)": 1})
    assert compat.compare(theirs, ours, adapted) == (
        Counter({"stats(n)": 1}),
        Counter({"count(n)": 1}),
        Counter(),
        Counter(),
    )
    # A line the PR changed since, which no adaptation names, is a difference on each side.
    changed = theirs - Counter({"b": 1}) + Counter({"b2": 1})
    assert compat.compare(changed, ours, adapted)[2:] == (Counter({"b2": 1}), Counter({"b": 1}))
    # An adaptation the PR no longer differs in is no difference either.
    assert compat.compare(ours, ours, adapted) == (Counter(), Counter(), Counter(), Counter())


def test_each_adaptation_is_under_a_backport_paragraph_of_its_patch(compat):
    """ADAPTED names an adaptation only under a commit that its patch's header gives a "Backport:" paragraph (as
    make-room-patches.sh writes it, indented under the commit's line), and each line it names as the backport's is
    one the patch adds."""
    for name, adaptations in compat.ADAPTED.items():
        text = next(p for p in compat.PATCHES if p.name == name).read_text()
        paragraphs, commit = {}, None
        for line in text.split("\ndiff --git ", 1)[0].splitlines():
            if match := re.match(r"  ([0-9a-f]{9}) ", line):
                commit = match[1]
            elif commit and line.startswith("    "):
                paragraphs[commit] = paragraphs.get(commit, "") + line.strip() + " "
        backports = {commit for commit, text in paragraphs.items() if text.startswith("Backport")}
        assert set(adaptations) <= backports, name
        lines = compat.added_lines(text)
        for commit, (_, ours) in adaptations.items():
            assert Counter(ours) <= lines, commit


@needs_merge_tree
def test_a_head_the_backport_does_not_follow_fails_naming_its_lines(tmp_path, upstream):
    """#63351 pushed again with a line the backport does not have fails the comparison, which names the line on each
    side. A head that took the backport's own form of an adapted line is no failure: the adaptation is stale."""
    old = "if self.current_step - start < self.priority_prefill_budget_cap_tokens\n"
    assert old in pr_lines(63351)
    base = upstream.commit(None, BASE | scheduler_files(real=False), "base")
    main = upstream.commit(base, {"vllm/main_only.py": "m = 1\n"}, "main moves on")
    kv = upstream.commit(base, KV, "#63350")
    pushed, adopted = (
        upstream.commit(
            kv, {f"{SCHED}/scheduler.py": CAP[f"{SCHED}/scheduler.py"].replace(*change)}
        )
        for change in (
            (old, old.replace(" < ", " <= ")),
            (
                "queue.has_lower_priority_than(request.priority)\n",
                "self._has_lower_priority_than(queue, request.priority)\n",
            ),
        )
    )
    proc = run(
        tmp_path / "pushed", upstream.serve(main, {63350: kv, 63351: pushed}, {"v0.40.0": base})
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert (
        "  #63351 and 63351.patch: 155 of #63351's 167 lines are the backport's; its other 11 and 14 of the "
        "backport's 170 are adapted (6511bc916, f1614decc); it DIFFERS in 2 lines no Backport paragraph names:\n"
        f"    #63351's: {old.replace(' < ', ' <= ')}"
        f"    63351.patch's: {old}"
    ) in proc.stdout
    assert (
        "backport DIFFERS from the PRs' heads in 2 lines no Backport paragraph names\n"
        in proc.stdout
    )
    proc = run(
        tmp_path / "adopted", upstream.serve(main, {63350: kv, 63351: adopted}, {"v0.40.0": base})
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "(6511bc916, f1614decc); 2 of the lines named as adapted no longer differ\n" in proc.stdout
    )
    assert (
        "backport = the PRs' heads but for 14 adapted lines of theirs and 14 of its own"
        in proc.stdout
    )
