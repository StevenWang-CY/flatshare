"""The window sweep (src/leaseholder_vllm/make-room/knee.sh), run in a copy of the deployed tree against stand-ins for
variant.sh, the engine's /server_info (curl), journalctl and tail.py: the measurement drop-in on, each window's switch
and check, its runs, a resume after a failed run, a refusal when the engine does not run the window, and the frozen
engine at the end."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

KNEE = Path(__file__).resolve().parents[1] / "src/leaseholder_vllm/make-room/knee.sh"
ROOM = "python/leaseholder_vllm/src/leaseholder_vllm/make-room"
"""knee.sh's directory in the deployed tree."""

FAKE_VARIANT = """\
#!/usr/bin/env bash
# variant.sh measure on|off, or use NAME [FLAG...]: records the call; a switch, as the engine would run it.
echo "$*" >>"$FAKE_STATE/variant.log"
[[ $1 == measure ]] || printf '%s\\n' "$@" >"$FAKE_STATE/engine"
"""

FAKE_CURL = """\
import json, os, sys
# curl -fsS URL/server_info?config_format=json: the additional_config the last switch builds, as vLLM reports it.
words = open(os.path.join(os.environ["FAKE_STATE"], "engine")).read().split()
config = {}
if words[1] == "make-room":
    room = {"max_preemptions_per_step": 2, "budget_cap": True, "reserve_tokens": 2048}
    flags = words[2:] if not os.environ.get("FAKE_IGNORE_FLAGS") else []
    for flag, value in zip(flags[::2], flags[1::2]):
        key = flag.removeprefix("--additional-config.make_room.")
        room[key] = {"true": True, "false": False}.get(value, int(value) if value.isdigit() else value)
    config = {"make_room": room}
print(json.dumps({"vllm_config": {"additional_config": config}}))
"""

FAKE_TAIL = """\
import os, sys
# tail.py ENGINE SECONDS AGENTS MODE --reply TOKENS: records the run, fails the FAKE_TAIL_FAIL-th, prints a summary.
state = os.environ["FAKE_STATE"]
with open(os.path.join(state, "tail.log"), "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
runs = sum(1 for _ in open(os.path.join(state, "tail.log")))
print("tail: 12 agents with fresh prompts of 6,000-30,000 tokens beside 2 calls with 40-token replies")
if str(runs) == os.environ.get("FAKE_TAIL_FAIL"):
    print("Traceback: the engine went away")
    sys.exit(1)
print("calls: 2 turns, 0 failed; agents: 0 of 12 stopped early")
"""


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    """A deployed source tree with knee.sh and the stand-ins, a home with the engine's python, and a state directory."""
    src, home, state, bin_dir = (tmp_path / name for name in ("src", "home", "state", "bin"))
    room = src / ROOM
    room.mkdir(parents=True)
    (src / "deploy/phastform").mkdir(parents=True)
    shutil.copy(KNEE, room / "knee.sh")
    (room / "tail.py").write_text(FAKE_TAIL)
    (src / "deploy/phastform/variant.sh").write_text(FAKE_VARIANT)
    (src / "DEPLOYED").write_text("abc1234 2026-10-25T00:00:00Z\n")
    engine_bin = home / "leaseholder/venvs/engine/bin"
    engine_bin.mkdir(parents=True)
    (engine_bin / "python").symlink_to(sys.executable)
    state.mkdir()
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(f"#!{sys.executable}\n{FAKE_CURL}")
    (bin_dir / "journalctl").write_text(
        "#!/bin/sh\necho 'make-room: the budget cap held prefill back'\n"
    )
    for fake in ("curl", "journalctl"):
        (bin_dir / fake).chmod(0o755)
    return {"room": room, "home": home, "state": state, "bin": bin_dir, "run": tmp_path / "run"}


def knee(tree: dict[str, Path], *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    environ = (
        os.environ
        | env
        | {
            "HOME": str(tree["home"]),
            "PATH": f"{tree['bin']}{os.pathsep}{os.environ['PATH']}",
            "FAKE_STATE": str(tree["state"]),
            "XDG_RUNTIME_DIR": str(tree["run"]),
        }
    )
    return subprocess.run(
        ["bash", str(tree["room"] / "knee.sh"), "--seconds", "5", *args],
        env=environ,
        capture_output=True,
        text=True,
    )


def lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def test_each_window_switches_checks_and_runs(tree):
    done = knee(tree, "16", "48", "off")
    assert done.returncode == 0, done.stdout + done.stderr
    flags = "use make-room --additional-config.make_room.reserve_tokens 0"
    assert lines(tree["state"] / "variant.log") == [
        "measure on",
        f"{flags} --additional-config.make_room.budget_cap_tokens 16",
        f"{flags} --additional-config.make_room.budget_cap_tokens 48",
        f"{flags} --additional-config.make_room.budget_cap false",
        "use frozen",
    ]
    engine = f"unix:{tree['run']}/lh/engine.sock"
    assert lines(tree["state"] / "tail.log") == [f"{engine} 5 pool fresh --reply 40"] * 6
    out = tree["home"] / "leaseholder/runs/window-knee"
    for window in ("16", "48", "off"):
        for run in (1, 2):
            assert lines(out / f"w{window}-{run}.out")[-1].startswith("calls: 2 turns")
        assert lines(out / f"w{window}-make-room.log") == [
            "make-room: the budget cap held prefill back"
        ]
    assert json.loads((out / "w48-engine.json").read_text())["make_room"]["budget_cap_tokens"] == 48
    assert json.loads((out / "woff-engine.json").read_text())["make_room"]["budget_cap"] is False
    assert lines(out / "DEPLOYED") == ["abc1234 2026-10-25T00:00:00Z"]
    assert "knee: done" in lines(out / "knee.log")[-2]


def test_a_failed_run_stops_the_sweep_and_a_rerun_resumes_it(tree):
    # The third run (window 48's first) fails: the box goes back to the frozen engine, and window 16 stays complete.
    failed = knee(tree, "--reply", "32", "--mode", "grow", "16", "48", FAKE_TAIL_FAIL="3")
    assert failed.returncode == 1
    assert lines(tree["state"] / "variant.log")[-1] == "use frozen"
    out = tree["home"] / "leaseholder/runs/window-knee"
    assert not (out / "w48-1.out").exists() and (out / "w48-1.part").exists()
    assert any("window 48: run 1 failed" in line for line in lines(out / "knee.log"))
    (tree["state"] / "variant.log").unlink()
    (tree["state"] / "tail.log").unlink()
    resumed = knee(tree, "--reply", "32", "--mode", "grow", "16", "48")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    # Window 16 is skipped without a switch; window 48 runs both.
    assert lines(tree["state"] / "variant.log") == [
        "measure on",
        "use make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48",
        "use frozen",
    ]
    engine = f"unix:{tree['run']}/lh/engine.sock"
    assert lines(tree["state"] / "tail.log") == [f"{engine} 5 pool grow --reply 32"] * 2
    assert any("window 16: complete" in line for line in lines(out / "knee.log"))


def test_an_engine_without_the_window_is_refused(tree):
    refused = knee(tree, "24", FAKE_IGNORE_FLAGS="1")
    assert refused.returncode == 1
    log = "\n".join(lines(tree["home"] / "leaseholder/runs/window-knee/knee.log"))
    assert "window 24: the engine runs make_room" in log and "budget_cap_tokens" in log
    assert not (tree["state"] / "tail.log").exists()
    assert lines(tree["state"] / "variant.log")[-1] == "use frozen"


def test_a_bad_argument_is_refused(tree):
    for args in (("sixteen",), ("--mode", "slow"), ("--runs",), ("--seconds", "0")):
        done = knee(tree, *args)
        assert done.returncode == 2 and "usage: knee.sh" in done.stderr, args
