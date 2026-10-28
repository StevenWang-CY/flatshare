# The change against vLLM's main and its next release (2026-10-27)

`make compat-upstream` ([`compat_upstream.py`](../../../python/flatshare_vllm/scripts/compat_upstream.py)) asks
whether #63350 and #63351 still merge into vLLM's main and pass their tests there, and whether make-room's patches
apply to the newest release. [`compat.log`](compat.log) is its output of 2026-10-27, home directories written as
`/home/USER`, after the interpreter's versions and the run's exit status (0). It ran on Gray (Ubuntu 24.04, git 2.43: the check merges with `git merge-tree --write-tree`, which
peacock's git 2.34 lacks), with an interpreter for vLLM's main built from main's CPU requirements (torch 2.14.0+cpu,
transformers 5.20.2).

- **Upstream.** vLLM's main at 4d95f222e6 (2026-10-27). #63350 at ee4bd6ae3e and #63351 at 9013ed433b each merge into
  it cleanly, and on their merges the PRs' tests pass on the CPU: 51 for #63350 and 81 for #63351
  (`tests/v1/core/test_priority_make_room.py`).
- **The next release.** make-room's patches apply to v0.32.0, the version the box pins and deploys, and not to
  v0.33.0rc0: the first hunk of `63350.patch` no longer matches the scheduler at its line 53, and v0.33.0rc0's
  `scheduler.py` differs from v0.32.0's by 333 lines added and 41 removed.

vLLM had released nothing after 0.32.0 (2026-10-18) by 2026-10-28, on GitHub or on PyPI; v0.33.0rc0 is a release
candidate's tag, not a release. So no supported upgrade could be built and tested on: the variant stays on 0.32.0,
and its patches are rebased when 0.33.0 is released, starting at the hunk this check names.
