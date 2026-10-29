"""Whether the engine change still fits vLLM upstream: #63350 and #63351 against vLLM's moving main, make-room's
backport of them against their heads, and its patches against vLLM's newest release.

    python scripts/compat_upstream.py [--clone DIR] [--python TORCH_PYTHON] [--git GIT] [--remote URL]

It keeps a bare blobless clone of vLLM in DIR (``~/.cache/flatshare/vllm-upstream``, beside the Makefile's v0.32.0
checkout) and its sparse worktrees in DIR.work, fetches vLLM's main, the PRs' heads (``refs/pull/N/head``) and the
release tags into it, and reports:

- for each PR, whether main holds it already: vLLM squash-merges, so a commit of main whose subject ends with the
  PR's number, "(#N)", or a true merge, or a merge that would change nothing; and if not, whether it merges into main
  cleanly (``git merge-tree --write-tree``, git 2.38 or newer), or the files that conflict. A PR stacked on one that
  main holds carries that one's commits, so only its own changes are merged, from that PR's head (``--merge-base``,
  git 2.40);
- the backport, 63350.patch and 63351.patch, against the PRs' heads: the lines each PR adds to scheduler.py, from
  where it leaves the PR it is stacked on or main, against those its patch adds, each net of those it removes. Where
  they differ, the lines must be those ADAPTED names, each under the commit whose "Backport:" paragraph in the patch's
  header says how v0.32.0 made it differ;
- with ``--python``, an interpreter with CPU torch and the Python dependencies of vLLM's main: on each clean merge, the
  test files the PR adds or changes (``tests/**/test_*.py`` in its diff), run on CPU from the merge's source; a test
  that fails on main too is main's, not the PR's. Each step that runs the interpreter has ``--timeout`` seconds (900),
  and a PR's test run that takes longer fails its files. An interpreter that cannot import main's scheduler is
  reported as such, and runs nothing;
- make-room's patches on a sparse checkout of the newest release, and of a newer release candidate if there is one
  (``build.py make-room OUT --source-tree``): they apply, or the first hunk that does not; and then, as build.py
  takes a hunk only exactly at its line, each hunk's fit as ``git apply`` finds it, the patches in order, with all
  three lines of its context (``-C3``) and with one (``-C1``): the line it applies at and its offset, any context git
  left out, or that git rejects it, which leaves the hunks after it to apply to the rest.

Exit status 1 for a conflict, a failed test, a backport that differs from the PRs' heads in a line no Backport
paragraph names, or patches that do not apply to v0.32.0, the pinned version; 2 when it could not check (no network,
an older git, an interpreter that cannot import main's scheduler or collect its tests, a PR none of whose tests ran);
0 otherwise, also when a newer release does not take the patches: the variant is written against 0.32.0, the pinned
and deployed version, and says so.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REMOTE = "https://github.com/vllm-project/vllm"
PRS = (63350, 63351)
PINNED = "v0.32.0"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
BUILD = Path(__file__).resolve().parents[1] / "src/flatshare_vllm/build.py"
# build.py, for its table of make-room's patches, in the order it applies them, and its reading of a hunk's header.
spec = importlib.util.spec_from_file_location("flatshare_vllm_build", BUILD)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
PATCHES = [BUILD.parent / "make-room" / name for name in build.VARIANTS["make-room"][0]]
ADAPTED = {
    "63350.patch": {
        # v0.32.0's SchedulerStats has no field for make-room's preemptions: the scheduler counts them.
        "58ca55361": (
            (
                "# Requests preempted to make room since the last scheduler stats.",
                "num_priority_preemptions = self._num_priority_preemptions",
                "num_priority_preemptions=num_priority_preemptions,",
                "self._num_priority_preemptions = 0",
            ),
            ("# Requests preempted to make room.",),
        ),
    },
    "63351.patch": {
        # v0.32.0's request queues stay stock: the queues' new method is the scheduler's _has_lower_priority_than.
        "6511bc916": (
            ("queue.has_lower_priority_than(request.priority)",),
            (
                "PriorityRequestQueue,",
                "self._has_lower_priority_than(queue, request.priority)",
                "@staticmethod",
                "def _has_lower_priority_than(queue: RequestQueue, priority: int) -> bool:",
                '"""Whether a request in `queue` has a lower priority, a larger value,',
                "than `priority`: a priority queue's largest requests are among its",
                'heap\'s leaves, its second half."""',
                "if isinstance(queue, PriorityRequestQueue):",
                "heap = queue._heap",
                "return any(req.priority > priority for req in heap[len(heap) // 2 :])",
                "return any(req.priority > priority for req in queue)",
            ),
        ),
        # Nor for the budget cap's holds: the scheduler counts them too.
        "f1614decc": (
            (
                "# The budget cap's holds since the last scheduler stats: the steps in",
                "# which it held lower-priority prefill back, and the prefill tokens it",
                "# held back in them; and the last such step, with the tokens held back in",
                "# it so far.",
                "num_prefill_held_steps = self._num_prefill_held_steps",
                "num_prefill_held_steps=num_prefill_held_steps,",
                "num_prefill_held_tokens = self._num_prefill_held_tokens",
                "num_prefill_held_tokens=num_prefill_held_tokens,",
                "self._num_prefill_held_steps = 0",
                "self._num_prefill_held_tokens = 0",
            ),
            (
                "# The budget cap's holds: the steps in which it held lower-priority",
                "# prefill back, and the prefill tokens it held back in them; and the last",
                "# such step, with the tokens held back in it so far.",
            ),
        ),
    },
}
"""The lines in which make-room's backport of the PRs differs from their heads, by patch and by the commit whose
"Backport:" paragraph in that patch's header says how v0.32.0 made it differ: the PR's lines and the backport's, each
stripped, as the comparison counts them. The comparison holds every other line the PRs add to the patches'."""
RELEASE = re.compile(r"v(\d+(?:\.\d+)+)(?:rc(\d+))?")
# What a test run needs to import, from the source tree it runs on.
IMPORTS = "import pytest, torch, vllm.v1.core.sched.scheduler"
# Seconds each step that runs the interpreter may take: the import check, the collection of main's tests, and each PR's
# test run (the PRs' test files take 20 to 40 s on a 2.1 GHz Xeon).
TIMEOUT = 900


class CannotCheck(Exception):
    """The check could not run: the network, git or the interpreter, not the PRs."""


class Git:
    def __init__(self, git: str, clone: Path):
        self.git, self.clone = git, clone

    def __call__(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        proc = self.run(*args, cwd=cwd)
        if check and proc.returncode:
            raise CannotCheck(f"git {' '.join(args[:3])} failed: {last_line(proc.stderr)}")
        return proc.stdout

    def run(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        where = ["-C", str(cwd or self.clone)]
        return subprocess.run([self.git, *where, *args], capture_output=True, text=True)


def last_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def git_version(git: str) -> tuple[int, ...]:
    try:
        out = subprocess.run([git, "version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise CannotCheck(f"no git at {git!r}: {error}") from error
    match = re.search(r"(\d+)\.(\d+)", out)
    return tuple(map(int, match.groups())) if match else (0, 0)


def fetch(git: Git, remote: str) -> None:
    """The clone, made on first use, with main, the PRs' heads and the release tags fetched into it."""
    offline = (
        f"cannot fetch from {remote}: compat-upstream fetches vLLM's main, the PRs' heads and its release tags "
        "from there, so it needs the network"
    )
    if not (git.clone / "HEAD").exists():
        git.clone.parent.mkdir(parents=True, exist_ok=True)
        cmd = [git.git, "clone", "-q", "--bare", "--single-branch", "--branch", "main"]
        cmd += ["--filter=blob:none", remote, str(git.clone)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise CannotCheck(f"{offline} ({last_line(proc.stderr)})")
    git("config", "remote.origin.url", remote)
    refspecs = ["+refs/heads/main:refs/remotes/origin/main", "+refs/tags/v*:refs/tags/v*"]
    refspecs += [f"+refs/pull/{pr}/head:refs/remotes/origin/pull/{pr}" for pr in PRS]
    proc = git.run("fetch", "-q", "--filter=blob:none", "origin", *refspecs)
    if proc.returncode:
        raise CannotCheck(f"{offline} ({last_line(proc.stderr)})")


def merge(git: Git, pr: int, base: str | None = None) -> tuple[str | None, list[str]]:
    """The tree of the PR merged into main, or None and the files that conflict. With ``base``, the head of a PR it is
    stacked on that main holds, only the PR's own changes, from that head on, are merged (git 2.40's --merge-base)."""
    args = ["merge-tree", "--write-tree", "--name-only", "--no-messages"]
    if base:
        if git_version(git.git) < (2, 40):
            raise CannotCheck(
                f"#{pr} is stacked on a PR main holds, and merging its own changes needs git merge-tree "
                f"--merge-base, git 2.40 or newer"
            )
        args.append(f"--merge-base={base}")
    proc = git.run(*args, "origin/main", f"origin/pull/{pr}")
    lines = proc.stdout.splitlines()
    if proc.returncode == 0:
        return lines[0], []
    if proc.returncode == 1:
        return None, sorted(set(lines[1:]) - {""})
    raise CannotCheck(f"git merge-tree failed for #{pr}: {last_line(proc.stderr)}")


def landed(git: Git, pr: int) -> str | None:
    """How main holds the PR, if it does. vLLM squash-merges a PR into one commit whose subject ends with its number,
    "(#N)", so its head never becomes main's ancestor; a true merge, which makes it one, is told too. None if main
    holds neither."""
    head = f"origin/pull/{pr}"
    if git.run("merge-base", "--is-ancestor", head, "origin/main").returncode == 0:
        return "merged into main"
    base = git("merge-base", "origin/main", head).strip()
    for line in git("log", "--format=%h %s", "--abbrev=10", f"{base}..origin/main").splitlines():
        sha, _, subject = line.partition(" ")
        if subject.rstrip().endswith(f"(#{pr})"):
            return f"squash-merged into main as {sha}"
    return None


def changed_tests(git: Git, pr: int, base: str | None = None) -> list[str]:
    """The test files the PR adds or changes, against where it leaves main, or ``base``."""
    base = base or git("merge-base", "origin/main", f"origin/pull/{pr}").strip()
    out = git(
        "diff", "--name-only", "--diff-filter=AM", base, f"origin/pull/{pr}", "--",
        ":(glob)tests/**/test_*.py",
    )  # fmt: skip
    return out.split()


def worktree(git: Git, name: str, rev: str, paths: list[str]) -> Path:
    """A sparse worktree of ``rev`` with ``paths``, in the clone's work directory, made on first use."""
    tree = git.clone.with_name(git.clone.name + ".work") / name
    if not (tree / ".git").exists():
        git("worktree", "prune")
        tree.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "-q", "--detach", "--no-checkout", str(tree), rev)
    git("sparse-checkout", "set", "--cone", *paths, cwd=tree)
    git("checkout", "-q", "--detach", "--force", rev, cwd=tree)
    return tree


def source_env(tree: Path) -> dict[str, str]:
    return os.environ | {
        "PYTHONPATH": str(tree),
        "VLLM_TARGET_DEVICE": "cpu",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def can_import(python: str, tree: Path, timeout: float = TIMEOUT) -> str | None:
    """None if ``python`` imports pytest, torch and the scheduler from ``tree``, or why not."""
    try:
        proc = subprocess.run(
            [python, "-c", IMPORTS],
            cwd=tree,
            env=source_env(tree),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except OSError as error:
        return str(error)
    except subprocess.TimeoutExpired:
        return f"the import timed out after {timeout:g} s"
    return None if proc.returncode == 0 else last_line(proc.stderr)


def harness_error(python: str, tree: Path, dirs: list[str], timeout: float = TIMEOUT) -> str | None:
    """None if pytest collects ``dirs`` from ``tree``: its conftest files and test modules import. Or why not."""
    try:
        proc = subprocess.run(
            [python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *dirs],
            cwd=tree,
            env=source_env(tree),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"pytest's collection timed out after {timeout:g} s"
    if proc.returncode in (0, 5):  # 5: no tests collected
        return None
    output = proc.stdout + proc.stderr
    errors = [line[1:].strip() for line in output.splitlines() if line.startswith("E ")]
    return errors[-1] if errors else last_line(output)


def run_tests(
    python: str, tree: Path, files: list[str], timeout: float = TIMEOUT
) -> tuple[int, set[str], str]:
    """``files`` run with pytest from ``tree``, on CPU: how many tests passed (not failed, erred or skipped), the node
    IDs of those that failed or erred, and pytest's summary line. A run that takes more than ``timeout`` seconds is
    stopped and its files fail."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report}"]
        cmd += ["-o", "junit_family=xunit1", *files]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tree,
                env=source_env(tree),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            late = f"timed out after {timeout:g} s"
            return 0, {f"{' '.join(files)}: {late}"}, late
        if not report.exists():
            return (
                0,
                {f"{' '.join(files)}: pytest did not run"},
                last_line(proc.stdout + proc.stderr),
            )
        cases = list(ET.parse(report).iter("testcase"))
        failed = {
            node_id(case)
            for case in cases
            if case.find("failure") is not None or case.find("error") is not None
        }
        skipped = sum(case.find("skipped") is not None for case in cases)
    passed = len(cases) - len(failed) - skipped
    summary = next(
        (
            line
            for line in reversed(proc.stdout.splitlines())
            if re.search(r"\d+ (passed|failed|errors?|skipped)", line)
        ),
        last_line(proc.stdout),
    )
    return passed, failed, summary.strip("= ")


def node_id(case: ET.Element) -> str:
    """A JUnit test case's pytest node ID: its file, its class if any, its name."""
    file, classname, name = case.get("file"), case.get("classname") or "", case.get("name")
    if not file:  # a module that failed to collect
        return name
    module = file.removesuffix(".py").replace("/", ".")
    cls = (
        classname[len(module) + 1 :].replace(".", "::")
        if classname.startswith(module + ".")
        else ""
    )
    return "::".join(part for part in (file, cls, name) if part)


def newest(tags: list[str]) -> list[str]:
    """Of ``tags``, the newest release and, if one is newer, the newest release candidate."""
    finals, candidates = [], []
    for tag in tags:
        match = RELEASE.fullmatch(tag)
        if match:
            version = tuple(map(int, match[1].split(".")))
            if match[2] is None:
                finals.append((version, tag))
            else:
                candidates.append((version, int(match[2]), tag))
    if not finals:
        return []
    final = max(finals)
    newer = [c for c in candidates if c[0] > final[0]]
    return [final[1]] + ([max(newer)[2]] if newer else [])


def builds(git: Git, tag: str) -> str | None:
    """None if make-room builds on ``tag``'s scheduler, or build.py's reason: the first hunk that does not apply."""
    tree = worktree(git, "release", tag, ["vllm/v1/core/sched"])
    with tempfile.TemporaryDirectory() as out:
        proc = subprocess.run(
            [sys.executable, str(BUILD), "make-room", out, "--source-tree"],
            env=os.environ | {"PYTHONPATH": str(tree)},
            capture_output=True,
            text=True,
        )
    return None if proc.returncode == 0 else last_line(proc.stderr).removeprefix("build.py: ")


def hunk_fits(report: str, headers: list[str]) -> list[str] | None:
    """Each hunk's fit as ``git apply --reject --verbose`` reports it, in the C locale, for a patch whose hunks have
    ``headers``: "at L", the line it applied at, with the offset git found it at and the context it kept where they are
    not the hunk's own; or "rejected". None if the report accounts for not every hunk, as when git applied nothing."""
    every = set(range(1, len(headers) + 1))
    rejected = {int(n) for n in re.findall(r"^Rejected hunk #(\d+)\.$", report, re.M)}
    if re.search(r"^Applied patch .+ cleanly\.$", report, re.M):
        applied = every
    else:
        applied = {int(n) for n in re.findall(r"^Hunk #(\d+) applied cleanly\.$", report, re.M)}
    if applied | rejected != every or applied & rejected:
        return None
    moved = {
        int(n): (int(at), int(offset))
        for n, at, offset in re.findall(
            r"^Hunk #(\d+) succeeded at (\d+) \(offset (-?\d+) lines?\)\.$", report, re.M
        )
    }
    # git reports no line for a hunk it found where the hunk says it is: at its new start.
    at = {n: moved.get(n, (int(build.HUNK.match(headers[n - 1])[3]), 0)) for n in applied}
    found = {
        n: f"at {line}" + (f" (offset {offset} lines)" if offset else "")
        for n, (line, offset) in at.items()
    }
    # git names a hunk it kept less context for by the line it applied at, which a moved hunk's line gives too: the
    # hunk that applied nearest to it.
    for leading, trailing, line in re.findall(
        r"^Context reduced to \((\d+)/(\d+)\) to apply fragment at (\d+)$", report, re.M
    ):
        n = min(applied, key=lambda n: abs(at[n][0] - int(line)))
        found[n] += f", context reduced to {leading}/{trailing}"
    return [found.get(n, "rejected") for n in sorted(every)]


def fits(git: str, source: str, context: int, patches: list[Path]) -> list[tuple[str, str]]:
    """Each hunk of ``patches``, by its patch, number and header, with its fit when git apply applies them in order to
    ``source``, a scheduler.py, with at least ``context`` lines of context matching on each side of a change. A hunk
    git rejects is left out, and the next ones apply to the rest, as a rebase would start."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # A repository of its own, so that git apply reads the patches' paths from here, whatever encloses it.
        proc = subprocess.run([git, "init", "-q", tmp], capture_output=True, text=True)
        if proc.returncode:
            raise CannotCheck(f"git init failed: {last_line(proc.stderr)}")
        (root / SCHEDULER).parent.mkdir(parents=True)
        (root / SCHEDULER).write_text(source)
        rows = []
        for patch in patches:
            headers = [m[0] for m in map(build.HUNK.match, patch.read_text().splitlines()) if m]
            proc = subprocess.run(
                [git, "apply", "--reject", "--verbose", f"-C{context}", str(patch)],
                cwd=root,
                env=os.environ | {"LC_ALL": "C"},
                capture_output=True,
                text=True,
            )
            found = hunk_fits(proc.stderr, headers)
            if found is None:
                raise CannotCheck(f"git apply -C{context} {patch.name}: {last_line(proc.stderr)}")
            labels = [f"{patch.name} #{n} {header}" for n, header in enumerate(headers, 1)]
            rows += zip(labels, found, strict=True)
    return rows


def added_lines(diff: str) -> Counter[str]:
    """The lines a diff of scheduler.py adds, net of those it removes, so that a line it moves counts in neither: each
    stripped, blank ones aside, as the backport is compared with the PRs. A patch's header, before its diff, is none."""
    added, removed = Counter(), Counter()
    hunks = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            hunks = False
        elif line.startswith("@@"):
            hunks = True
        elif hunks and line.startswith(("+", "-")) and line[1:].strip():
            (added if line[0] == "+" else removed)[line[1:].strip()] += 1
    return added - removed


def compare(
    theirs: Counter[str], ours: Counter[str], adapted: dict[str, tuple[tuple[str, ...], ...]]
) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str]]:
    """A PR's lines (``theirs``) against its patch's (``ours``): of the lines each has and the other lacks, those that
    ``adapted`` names, the PR's then the patch's, and those it does not, the PR's then the patch's."""
    named = [Counter(line for lines in adapted.values() for line in lines[side]) for side in (0, 1)]
    only = (theirs - ours, ours - theirs)
    return only[0] & named[0], only[1] & named[1], only[0] - named[0], only[1] - named[1]


def against_heads(git: Git) -> bool:
    """Prints how make-room's backport of the PRs, 63350.patch and 63351.patch, differs from the PRs' heads in the lines
    each adds to scheduler.py, and returns whether it differs in a line that no "Backport:" paragraph of their headers
    names (ADAPTED). A PR is compared by its own changes: from where it leaves the PR it is stacked on, or main."""
    print("the backport against the PRs' heads, by the lines each adds to scheduler.py:")
    patches = {patch.name: patch for patch in PATCHES}
    base, unnamed, adapted = "origin/main", 0, [0, 0]
    for pr in PRS:
        head, name = f"origin/pull/{pr}", f"{pr}.patch"
        if base != "origin/main" and git.run("merge-base", "--is-ancestor", base, head).returncode:
            base = "origin/main"  # rebased off the PR before it
        start = git("merge-base", base, head).strip()
        if base == "origin/main" and start == git("rev-parse", head).strip():
            print(f"  #{pr}: main holds its head, which leaves no base to compare its changes from")
            base = head
            continue
        theirs = added_lines(git("diff", start, head, "--", SCHEDULER))
        ours = added_lines(patches[name].read_text())
        named_theirs, named_ours, other_theirs, other_ours = compare(theirs, ours, ADAPTED[name])
        named = sum(named_theirs.values()) + sum(named_ours.values())
        listed = sum(len(lines) for both in ADAPTED[name].values() for lines in both)
        line = (
            f"  #{pr} and {name}: {sum((theirs & ours).values())} of #{pr}'s {sum(theirs.values())} lines are "
            f"the backport's; its other {sum(named_theirs.values())} and {sum(named_ours.values())} of the "
            f"backport's {sum(ours.values())} are adapted ({', '.join(ADAPTED[name])})"
        )
        if named < listed:
            line += f"; {listed - named} of the lines named as adapted no longer differ"
        differs = sum(other_theirs.values()) + sum(other_ours.values())
        print(
            line
            + (f"; it DIFFERS in {differs} lines no Backport paragraph names:" if differs else "")
        )
        for whose, lines in ((f"#{pr}'s", other_theirs), (f"{name}'s", other_ours)):
            shown = sorted(lines.elements())
            for text in shown[:10]:
                print(f"    {whose}: {text}")
            if len(shown) > 10:
                print(f"    and {len(shown) - 10} more of {whose}")
        unnamed += differs
        adapted = [adapted[0] + sum(named_theirs.values()), adapted[1] + sum(named_ours.values())]
        base = head
    if unnamed:
        print(
            f"backport DIFFERS from the PRs' heads in {unnamed} lines no Backport paragraph names"
        )
    else:
        print(
            f"backport = the PRs' heads but for {adapted[0]} adapted lines of theirs and {adapted[1]} of its own, "
            "which the patches' headers name"
        )
    return bool(unnamed)


def check(
    clone: Path, python: str | None, git_cmd: str, remote: str, timeout: float = TIMEOUT
) -> int:
    if git_version(git_cmd) < (2, 38):
        raise CannotCheck(
            f"git merge-tree --write-tree needs git 2.38 or newer, and {git_cmd} is "
            f"{'.'.join(map(str, git_version(git_cmd)))}: GIT=<a newer git> points the target at one"
        )
    git = Git(git_cmd, clone)
    fetch(git, remote)
    heads = {pr: git("rev-parse", "--short=10", f"origin/pull/{pr}").strip() for pr in PRS}
    main = git("rev-parse", "--short=10", "origin/main").strip()
    date = git("log", "-1", "--format=%cs", "origin/main").strip()
    print(f"vLLM main {main} ({date}); " + ", ".join(f"#{pr} {heads[pr]}" for pr in PRS))

    failed = False
    merges = {}
    main_tree_id = git("rev-parse", "origin/main^{tree}").strip()
    held: dict[int, str] = {}  # the PRs main holds, by their heads
    for pr in PRS:
        head = git("rev-parse", f"origin/pull/{pr}").strip()
        how = landed(git, pr)
        if how:
            print(f"#{pr}: {how}")
            held[pr] = head
            continue
        # A PR stacked on one main holds carries that one's commits: merge only its own changes.
        on = next(
            (
                n
                for n, h in held.items()
                if git.run("merge-base", "--is-ancestor", h, head).returncode == 0
            ),
            None,
        )
        tree, conflicts = merge(git, pr, held.get(on))
        subject, suffix = (
            (f"#{pr}: its own changes", f", on #{on} as main holds it") if on else (f"#{pr}:", "")
        )
        if tree is None:
            failed = True
            verb = "CONFLICT" if on else "CONFLICTS"
            print(f"{subject} {verb} with main in {', '.join(conflicts)}{suffix}")
        elif tree == main_tree_id:
            print(f"#{pr}: its changes are in main already")
        else:
            merges[pr] = (tree, held.get(on))
            print(f"{subject} {'merge' if on else 'merges'} into main cleanly{suffix}")
    failed |= against_heads(git)

    cannot = None
    if not python and merges:
        print("the PRs' tests: not run (TORCH_PYTHON=<an interpreter for vLLM main> runs them)")
    if python and merges:
        main_tree = worktree(git, "main", "origin/main", ["vllm", "tests"])
        tests = {pr: changed_tests(git, pr, base) for pr, (_, base) in merges.items()}
        dirs = {str(Path(f).parent) for files in tests.values() for f in files}
        dirs = sorted(d for d in dirs if (main_tree / d).is_dir())
        why = can_import(python, main_tree, timeout)
        if why:
            cannot = f"{python} cannot import vLLM main's scheduler on CPU ({why}): no test ran"
        elif dirs and (why := harness_error(python, main_tree, dirs, timeout)):
            cannot = (
                f"{python} cannot run vLLM main's tests in {', '.join(dirs)} ({why}): no test ran"
            )
        print(cannot or f"the PRs' tests on their merges into main, on CPU ({python}):")
        for pr, (tree_id, _) in ({} if cannot else merges).items():
            files = tests[pr]
            if not files:
                print(f"  #{pr}: no test file added or changed")
                continue
            commit = git(
                "-c", "user.name=compat-upstream", "-c", "user.email=compat-upstream@localhost",
                "commit-tree", tree_id, "-p", "origin/main", "-p", f"origin/pull/{pr}",
                "-m", f"#{pr} merged into main",
            ).strip()  # fmt: skip
            tree = worktree(git, f"merge-{pr}", commit, ["vllm", "tests"])
            passed, failures, summary = run_tests(python, tree, files, timeout)
            mains = set()
            on_main = [f for f in files if (main_tree / f).exists()]
            if failures and on_main:
                mains = run_tests(python, main_tree, on_main, timeout)[1] & failures
            line = f"  #{pr}: {', '.join(files)}: {summary}"
            if mains:
                line += f" ({len(mains)} of them fail on main too)"
            if not passed and not failures:
                # Every test skipped, a CUDA-only mark or an importorskip, say: that is no evidence.
                line += ": no test ran, so this checked nothing"
                cannot = cannot or f"no test of #{pr} ran"
            print(line)
            for name in sorted(failures - mains):
                failed = True
                print(f"    FAILED {name}")

    tags = newest(git("tag", "-l", "v*").split())
    if not tags:
        raise CannotCheck("no release tag fetched")
    for tag in tags:
        why = builds(git, tag)
        what = f"make-room's patches on {tag}"
        if tag == PINNED:
            print(
                f"{what}, the pinned version: " + ("they apply" if why is None else f"FAIL: {why}")
            )
            failed |= why is not None
        elif why is None:
            print(f"{what}: they apply")
        else:
            diff = git.run("diff", "--shortstat", PINNED, tag, "--", SCHEDULER)
            stat = f"; its scheduler.py against {PINNED}: {diff.stdout.strip() or 'unchanged'}"
            print(
                f"{what}: do not apply ({why}){stat if diff.returncode == 0 else ''}. "
                f"The variant is written against {PINNED}, the pinned and deployed version"
            )
        if why is not None:
            # build.py takes a hunk only at its line, exactly, and stops at the first it cannot; git apply, which
            # looks for a hunk elsewhere and with less context, says how far each is from fitting.
            release = worktree(git, "release", tag, ["vllm/v1/core/sched"]) / SCHEDULER
            if not release.is_file():
                print(f"  no hunk can fit: {tag} has no {SCHEDULER}")
                continue
            three, one = (fits(git.git, release.read_text(), c, PATCHES) for c in (3, 1))
            took = [sum(fit != "rejected" for _, fit in rows) for rows in (three, one)]
            print(
                "  each hunk by git apply, the patches in order, with 3 lines of context and with 1: "
                f"{took[0]} and {took[1]} of {len(one)} apply"
            )
            for (hunk, with_three), (_, with_one) in zip(three, one, strict=True):
                print(f"    {hunk}: -C3 {with_three}; -C1 {with_one}")
    if failed:
        return 1
    if cannot:
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default = Path.home() / ".cache/flatshare/vllm-upstream"
    parser.add_argument(
        "--clone", type=Path, default=default, help=f"the clone (default {default})"
    )
    parser.add_argument(
        "--python", help="an interpreter with CPU torch and vLLM main's dependencies"
    )
    parser.add_argument("--git", default="git")
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT,
        help=f"seconds each interpreter step may take (default {TIMEOUT})",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    try:
        sys.exit(check(args.clone, args.python, args.git, args.remote, args.timeout))
    except CannotCheck as error:
        print(f"compat-upstream: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
