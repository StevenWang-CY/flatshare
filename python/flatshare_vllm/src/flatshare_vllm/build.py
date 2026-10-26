"""Build vLLM 0.32.0's scheduler with an engine variant's change, as an importable module for ``--scheduler-cls``.

``python build.py VARIANT OUT_DIR [--source-tree]`` reads the installed vLLM's ``vllm/v1/core/sched/scheduler.py`` and
``async_scheduler.py``, applies the variant's patches to the scheduler, appends the stock ``AsyncScheduler``, and
writes ``OUT_DIR/flatshare_<variant>_scheduler.py``. The engine then runs with ``PYTHONPATH=OUT_DIR`` and
``--scheduler-cls flatshare_<variant>_scheduler.AsyncScheduler``; the installed vLLM is never modified. Each variant's
``flatshare-engine.conf`` (deploy/phastform/variants) runs this at every engine start, with the engine's Python: it
needs only the standard library, and finding vLLM through ``importlib.util.find_spec`` does not import it, so it runs
without torch or a GPU.

The variants, one directory of patches each beside this file:

- ``make-room``: vLLM #63350 and #63351 backported to v0.32.0 (``63350.patch``, ``63351.patch``), then what only the
  variant has (``variant.patch``): the switches from ``--additional-config``'s ``make_room`` table, the reserve, and
  the counters' log line. Each patch is the ``scheduler.py`` part of a branch of commits on v0.32.0 in a vLLM clone,
  the PRs' commits cherry-picked and, where v0.32.0 needs it, adapted; ``scripts/make-room-patches.sh`` writes them,
  and a patch's header names its commits and says how any was adapted.
- ``pr54530``: vLLM PR #54530, ported to v0.32.0 by hand (``pr54530.patch``).

Every hunk must apply at its stated line, with its context and removed lines exactly as they are there: the patches
match 0.32.0's source line for line, so the build refuses any other installed version, and a source that differs
fails it, naming the hunk. ``--source-tree`` lets the tests build from a v0.32.0 source checkout, which has no package
metadata to check.

The change targets ``Scheduler.schedule``, which vLLM's ``AsyncScheduler`` inherits, and the frozen box schedules
asynchronously (vLLM's default for its configuration). So the module also defines ``AsyncScheduler``: the stock class,
appended after the patched ``Scheduler``, extends it instead of importing the stock one, and the variant loads that
one. With and without the change, the module differs by the change alone.

The generated module is vLLM's code (Apache-2.0, Copyright contributors to the vLLM project; its SPDX lines are kept)
with the change; this script and the patches, like the rest of the repository, are Apache-2.0.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

VLLM_VERSION = "0.32.0"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
HERE = Path(__file__).resolve().parent

# Each variant's patches, in the order they apply, and what its module says it is.
VARIANTS: dict[str, tuple[tuple[str, ...], str]] = {
    "make-room": (
        ("63350.patch", "63351.patch", "variant.patch"),
        "Flatshare's make-room (#63350, #63351)",
    ),
    "pr54530": (("pr54530.patch",), "vLLM PR #54530 ported to v0.32.0"),
}

# The stock AsyncScheduler, appended after the patched Scheduler, extends it instead of importing the stock one, and
# keeps the scheduler's logger, which its own would rebind to the module's name, outside vLLM's logging config.
ASYNC_EDITS = (
    ("from vllm.v1.core.sched.scheduler import Scheduler\n", ""),
    ("logger = init_logger(__name__)\n", ""),
)

HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    pass


def apply_patch(source: str, patch: str, name: str) -> str:
    """``source``, the stock scheduler.py, with ``patch`` applied: a unified diff of that file alone (text before its
    ``diff --git`` line is the patch's description). Each hunk applies at its stated line, its context and removed
    lines as they are there, or PatchError names it."""
    lines = source.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)
    starts = [i for i, line in enumerate(patch_lines) if line.startswith("diff --git ")]
    if len(starts) != 1 or patch_lines[starts[0]] != f"diff --git a/{SCHEDULER} b/{SCHEDULER}\n":
        raise PatchError(f"{name}: expected one diff, of {SCHEDULER}")
    i = starts[0] + 1
    while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
        if patch_lines[i].startswith(("--- ", "+++ ")) and patch_lines[i][4:].rstrip("\n") not in (
            f"a/{SCHEDULER}",
            f"b/{SCHEDULER}",
        ):
            raise PatchError(f"{name}: expected one diff, of {SCHEDULER}")
        i += 1
    out: list[str] = []
    pos = 0  # the next source line not yet copied, 0-based
    while i < len(patch_lines):
        header = patch_lines[i]
        match = HUNK.match(header)
        if not match:
            raise PatchError(f"{name}: expected a hunk at its line {i + 1}: {header!r}")
        old_start, old_len = int(match[1]), int(match[2] or 1)
        new_len = int(match[4] or 1)
        i += 1
        old: list[str] = []
        new: list[str] = []
        while len(old) < old_len or len(new) < new_len:
            if i == len(patch_lines):
                raise PatchError(f"{name}: hunk {header.strip()} ends early")
            line = patch_lines[i]
            if line.startswith("\\"):
                raise PatchError(
                    f"{name}: hunk {header.strip()}: the scheduler ends with a newline"
                )
            kind, text = line[:1], line[1:]
            if kind not in " -+":
                raise PatchError(f"{name}: hunk {header.strip()}: unexpected line {line!r}")
            if kind in " -":
                old.append(text)
            if kind in " +":
                new.append(text)
            i += 1
        start = old_start - 1 if old_len else old_start
        if start < pos or lines[start : start + old_len] != old:
            raise PatchError(
                f"{name}: hunk {header.strip()} does not match the scheduler at its line {start + 1}"
            )
        out += lines[pos:start] + new
        pos = start + old_len
    return "".join(out + lines[pos:])


def sched_dir(source_tree: bool) -> Path:
    """The scheduler package of the vLLM this interpreter finds, without importing vLLM (which needs a GPU stack): an
    installed vLLM 0.32.0, or with ``source_tree``, a source checkout, which has no package metadata to check."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("build.py: vllm is not installed in this interpreter")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        if not source_tree:
            sys.exit(
                "build.py: vllm has no package metadata; --source-tree builds from a source checkout"
            )
    else:
        if version != VLLM_VERSION:
            sys.exit(
                f"build.py: the change is written against vLLM {VLLM_VERSION}, and vLLM {version} is installed"
            )
    return Path(next(iter(spec.submodule_search_locations))) / "v1" / "core" / "sched"


def module_source(sched: Path, patches: Iterable[Path]) -> str:
    """The scheduler module's source without its header: ``sched``'s scheduler.py with ``patches`` applied in order,
    then its async_scheduler.py, whose AsyncScheduler extends the patched Scheduler."""
    source = (sched / "scheduler.py").read_text()
    for patch in patches:
        source = apply_patch(source, patch.read_text(), patch.name)
    async_source = (sched / "async_scheduler.py").read_text()
    for old, new in ASYNC_EDITS:
        if async_source.count(old) != 1:
            sys.exit(
                f"build.py: {old.strip()!r} found {async_source.count(old)} times in async_scheduler.py"
            )
        async_source = async_source.replace(old, new)
    return source + "\n\n" + async_source


def build(variant: str, out_dir: Path, source_tree: bool = False) -> Path:
    patches, what = VARIANTS[variant]
    name = f"flatshare_{variant.replace('-', '_')}_scheduler"
    try:
        source = module_source(sched_dir(source_tree), (HERE / variant / p for p in patches))
    except PatchError as error:
        sys.exit(f"build.py: {variant}: {error}")
    header = (
        f'"""vLLM v1 Scheduler and AsyncScheduler with {what},\n'
        'generated by python/flatshare_vllm/src/flatshare_vllm/build.py. Do not edit."""\n'
    )
    compile(header + source, f"{name}.py", "exec")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.py"
    # Whole or not at all: an engine that starts while this writes, or after a write was cut short, imports the
    # previous module or this one, never a part. The staged file's name is no module's.
    staged = out_dir / f".{name}.py.{os.getpid()}"
    try:
        staged.write_text(header + source)
        staged.replace(target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return target


if __name__ == "__main__":
    args = sys.argv[1:]
    source_tree = "--source-tree" in args
    args = [arg for arg in args if arg != "--source-tree"]
    if len(args) != 2 or args[0] not in VARIANTS:
        sys.exit(f"usage: build.py {{{','.join(VARIANTS)}}} OUT_DIR [--source-tree]")
    print(build(args[0], Path(args[1]), source_tree))
