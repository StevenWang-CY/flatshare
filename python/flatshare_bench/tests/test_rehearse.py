"""rehearse.sh on a CPU: the single-host rehearsal's every step against stand-ins for the hosts (directories that a fake
ssh runs commands in, with the real rsync between them), the box's user manager, nvidia-smi and ss, and a stub of the
kit's bootstrap.sh, run.sh and flatshare-svc shims; deploy/a100/push.sh and pull.sh are the repository's own. What it
holds: the checks refuse a box that runs, a busy GPU or a held port; nothing of the rehearsal reaches the box's
deployment (its source, its units, its user manager), and cleanup finds it as it was, or says what changed, once it has
ended what of the rehearsal runs, titled or not; the environment the kit runs in is the rehearsal's (its home, runtime
directory, flatshare-svc, the box's hub cache offline, no sudo); a command that fails ends its step, and a check that
gets no answer fails; a step on the host is detached, waited for and stopped with its whole group, and a step done is
not run again but under --redo (--purge runs cleanup again to remove the home); block 0's listener audit must pass with
each GPU server in a network namespace of its own (deploy/bin/in-private-net); block 4 is run for one point and its
deadline's stop; pull copies the rehearsal's records, not the box's; reproduce runs make on peacock over them.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HERE = REPO / "benchmarks" / "a100" / "rehearsal"
FAKES = REPO / "python" / "flatshare_bench" / "tests" / "fakes"
REHEARSE = HERE / "rehearse.sh"
HOST_SH = HERE / "host.sh"
QWEN, QWEN_REV = "Qwen/Qwen3-4B-Instruct-2507-FP8", "8591804019c8b22094c3b5b4454e0edc05dffc98"
WHISPER, WHISPER_REV = "openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

TOOLS = {
    # The box's user manager: it answers through the login session's bus, and not where no bus is named (the
    # rehearsal's environment; FAKE_NO_BUS sets what systemctl says there). Every call is logged: the test holds that
    # none acts on a unit.
    "systemctl": """\
#!/usr/bin/env bash
if [[ -z ${DBUS_SESSION_BUS_ADDRESS:-} ]]; then echo "${FAKE_NO_BUS-Failed to connect to bus: No such file or directory}" >&2; exit 1; fi
echo "systemctl $*" >>"$FAKE_BOX/.systemctl.log"
case "$*" in
  *is-system-running*) echo running ;;
  *list-unit-files*) cat "$FAKE_BOX/.unit-files" ;;
  *list-units*) cat "$FAKE_BOX/.units" ;;
esac
""",
    "nvidia-smi": """\
#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps*) cat "$FAKE_BOX/.gpu-apps" 2>/dev/null ;;
  *) echo "NVIDIA GeForce RTX 5060 Ti, 2 MiB, 16311 MiB" ;;
esac
""",
    "ss": """\
#!/usr/bin/env bash
case "$*" in
  *-Hltn*) cat "$FAKE_BOX/.listen-tcp" 2>/dev/null ;;
  *-Hlun*) cat "$FAKE_BOX/.listen-udp" 2>/dev/null ;;
esac
""",
    "unshare": "#!/bin/sh\nexit 0\n",
    # make on peacock: reproduce-a100 over the records it is given, as analysis.py writes its sections.
    "make": """\
#!/usr/bin/env bash
for a in "$@"; do case $a in RECORDS=*) records=${a#RECORDS=} ;; CORPUS=*) corpus=${a#CORPUS=} ;; esac; done
[[ " $* " == *" reproduce-a100 "* && -f $records/runs/a100/blocks.log && -d $corpus ]] || { echo "make: $*" >&2; exit 2; }
echo "make $*" >>"$HOME/make.log"
mkdir -p build/reproduce-a100
printf '%s\\n' "# The A100 40 GB run against its forecast" "" "## 1. The step model" "" "Not measured: no calibration point." \\
  "" "## 3. The probes" "" "- frozen: KV case 1650 ms" "" "## 4. E2 within the A100" "" "| point | v2v |" \\
  >build/reproduce-a100/a100.md
""",
}

STUB_BOOTSTRAP = """\
#!/usr/bin/env bash
# bootstrap.sh in the rehearsal's test: the environment it ran in, and what each stage leaves for rehearse.sh's checks.
set -euo pipefail
stage=$1
src=$(cd "$(dirname "$0")/../.." && pwd)
root=$HOME/flatshare
node=$root/node
mkdir -p "$node" "$root/cache"
{
  env | grep -E '^(HOME|XDG_RUNTIME_DIR|XDG_STATE_HOME|FLATSHARE_SUPERVISOR|HF_HUB_CACHE|HF_HUB_OFFLINE|DBUS_SESSION_BUS_ADDRESS|TMPDIR|FLATSHARE_REHEARSAL)=' | sort
  if sudo -n true 2>/dev/null; then echo "sudo worked"; else echo "sudo refused"; fi
} >"$root/bootstrap-$stage.env"
echo "bootstrap $stage: from $src"
[[ ${FAKE_FAIL:-} != "$stage" ]] || { echo "bootstrap $stage: failing, as the test asks"; exit 1; }
if [[ $stage == stage ]]; then
  printf '%s\\n' 'export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-'"$HOME"'/.local/state/flatshare-svc/run}' \\
    "export PATH=$src/deploy/bin/shims:\\$PATH" 'export FLATSHARE_SUPERVISOR=flatshare-svc' "export HF_HUB_CACHE=$HF_HUB_CACHE" >"$node/env.sh"
  source "$node/env.sh"
  systemctl --user start flatshare-voice-worker.service
fi
[[ $stage != warm ]] || echo '{"configurations": {"frozen": {"ok": true}, "pr": {"ok": true}}}' >"$root/cache/warm.json"
echo '{"counts": {"ok": 20, "degraded": 2, "refused": 0, "pending": 0}, "checks": [{"check": "Cores", "status": "degraded", "value": "16 CPUs"}]}' >"$node/preflight.json"
echo "bootstrap $stage: done (no check refused)"
"""

STUB_RUN = """\
#!/usr/bin/env bash
# run.sh in the rehearsal's test (it takes --until, --cap and --estimate BLOCK=MINUTES as the kit's, and skips a block
# done): the blocks' records, and block 0's listener audit as FAKE_AUDIT says.
set -euo pipefail
root=$HOME/flatshare
runs=$root/runs/a100
mkdir -p "$runs/blocks"
log() { echo "[$(date -u +%FT%TZ)] run: $*" | tee -a "$runs/blocks.log"; }
args="$*"
until="" estimate=""
while [[ ${1:-} == --* ]]; do
  case $1 in --until) until=$2 ;; --estimate) estimate=$2 ;; --cap) ;; esac
  shift 2
done
log "blocks $*${until:+, until $until UTC}"
for block in "$@"; do
  if [[ -f $runs/blocks/$block.done ]]; then
    log "block $block: done ($(cat "$runs/blocks/$block.done"))"
    continue
  fi
  case $block in
    0)
      mkdir -p "$runs/block0"
      printf '%-26s %s\\n' health ok "tts question (wav)" "0.3s" stt "0.4s: What is the capital of France?" llm "0.1s: Paris." \\
        "tts reply (first/total)" "0.2s/0.4s" "gpu memory" "13900 MiB, 16311 MiB" >"$runs/block0/smoke.txt"
      own="flatshare-engine, flatshare-stt, flatshare-tts in network namespaces of their own, with 12 TCP or UDP listeners and 3 abstract sockets there"
      case ${FAKE_AUDIT:-private} in
        private) audit="ok: the box's units hold 5 Unix sockets, all in $XDG_RUNTIME_DIR, and 1 TCP or UDP listeners, all allowed; $own" ;;
        degraded) audit="ok: the box's units hold 5 Unix sockets, all in $XDG_RUNTIME_DIR, and 1 TCP or UDP listeners, all allowed; the GPU servers run in the host's network namespace (FLATSHARE_PRIVATE_NET=host) with 12 loopback TCP listeners and 3 abstract sockets" ;;
        refused)
          echo "audit-listeners: flatshare-engine.service listens on tcp 127.0.1.1:41234" >&2
          echo "smoke: the listener audit failed (above)" >&2
          exit 1 ;;
      esac
      printf '%-26s %s\\n' "listener audit" "$audit" >>"$runs/block0/smoke.txt"
      echo "smoke: ok" >>"$runs/block0/smoke.txt"
      echo "bots: 1 caller, 2 turns, 0 failed" >"$runs/block0/call.txt" ;;
    1)
      mkdir -p "$runs/block1/frozen-e0"
      echo '{}' >"$runs/block1/frozen-e0/summary.json"
      for f in kv-1 kv-2 kv-3 prefill-1 prefill-2 prefill-3 tail-1 tail-2; do echo "$f: priority-0 TTFT 1650 ms" >"$runs/block1/frozen-$f.txt"; done ;;
    4)
      [[ -n $until && $estimate == 4=15 ]] || { echo "run.sh: the rehearsal's block 4 wants --until and --estimate 4=15" >&2; exit 3; }
      echo "$args" >"$runs/block4.args"
      points=$root/results/e2/a100-r1
      mkdir -p "$points/w5-disc-pr-ab48-s24"
      echo '{"status": "complete"}' >"$points/w5-disc-pr-ab48-s24/run.json"
      echo "[t] w5-a1-voice: 9 minutes would end after $until UTC; stopping before it (--until)" >>"$points/campaign.log"
      log "a100-r1 stopped before its deadline ($until UTC) with points left"
      exit 0 ;;
  esac
  date -u +%FT%TZ >"$runs/blocks/$block.done"
  log "block $block: done"
done
"""

STUB_SYSTEMCTL = """\
#!/usr/bin/env bash
# deploy/bin/shims/systemctl in the rehearsal's test: flatshare-svc's units, their states in the rehearsal's state
# directory. FAKE_UNSTOPPABLE names a unit that its stop leaves deactivating, as flatshare-svc shows one while it ends
# its group's rest.
dir=${XDG_STATE_HOME:?}/fake-flatshare-svc
mkdir -p "$dir"
echo "systemctl $*" >>"$dir/calls"
words=()
for a in "$@"; do [[ $a == --* ]] || words+=("$a"); done
action=${words[0]}
for unit in "${words[@]:1}"; do
  unit=${unit%.service}
  case $action in
    start)
      echo active >"$dir/$unit"
      if [[ $unit == flatshare-voice-worker ]]; then for u in flatshare-voicenet flatshare-livekit flatshare-gateway; do echo active >"$dir/$u"; done; fi ;;
    stop) if [[ $unit == "${FAKE_UNSTOPPABLE:-}" ]]; then echo deactivating; else echo inactive; fi >"$dir/$unit" ;;
    is-active) cat "$dir/$unit" 2>/dev/null || echo inactive ;;
  esac
done
"""

STUB_JOURNALCTL = (
    "#!/usr/bin/env bash\n"
    'echo "1761400000.000000 phastform flatshare-engine[42]: a line of the engine\'s"\n'
)

# titled.py TITLE PIDS [HOME]: a process that sets TITLE, as AIPerf's controller and vLLM's engine core set theirs
# (setproctitle, which blanks /proc/PID/environ), and starts in its process group a child that sets one too and,
# given HOME, another whose environment names that home; the children's pids go to the file PIDS.
TITLED = """\
import os, subprocess, sys, time
import setproctitle

if sys.argv[1]:
    setproctitle.setproctitle(sys.argv[1])
title = "import setproctitle, time; setproctitle.setproctitle('aiperf worker_manager'); time.sleep(120)"
children = [subprocess.Popen([sys.executable, "-c", title])]
if len(sys.argv) > 3:
    env = os.environ | {"HOME": sys.argv[3]}
    children.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], env=env))
with open(sys.argv[2] + ".new", "w") as pids:
    pids.write(" ".join(str(child.pid) for child in children))
os.rename(sys.argv[2] + ".new", sys.argv[2])
time.sleep(120)
"""

PROC = pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="reads /proc (Linux)")


def write(path: Path, text: str, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


@pytest.fixture
def world(tmp_path: Path) -> dict:
    """The workstation's checkout and data, the box (phastform's login home with its deployment), cyan and peacock."""
    hosts = tmp_path / "hosts"
    box = hosts / "phastform"
    # The box's deployment, as the rehearsal must leave it.
    write(box / "flatshare" / "src" / "DEPLOYED", "v0.3-191-g987fd09\n")
    write(box / "flatshare" / "src" / "deploy" / "phastform" / "engine.yaml", "model: the box's\n")
    write(box / "flatshare" / "runs" / "box-only.txt", "a record of the box's\n")
    write(box / "flatshare" / "venvs" / "engine" / "bin" / "python", "#!/bin/sh\n", 0o755)
    kokoro = box / "flatshare" / "vendor" / "Kokoro-FastAPI"
    write(kokoro / ".git" / "HEAD", "ebd4ec9\n")
    write(kokoro / "api" / "src" / "models" / "v1_0" / "kokoro-v1_0.pth", "weights\n")
    (box / ".config" / "systemd" / "user").mkdir(parents=True)
    (box / ".config" / "systemd" / "user" / "flatshare-engine.service").symlink_to(
        box / "flatshare" / "src" / "deploy" / "phastform" / "engine.yaml"
    )
    hub = box / ".cache" / "huggingface" / "hub"
    for model, revision in ((QWEN, QWEN_REV), (WHISPER, WHISPER_REV)):
        write(
            hub / f"models--{model.replace('/', '--')}" / "snapshots" / revision / "config.json",
            "{}\n",
        )
    write(box / ".units", "flatshare-engine.service loaded inactive dead Flatshare LLM engine\n")
    write(
        box / ".unit-files",
        "flatshare-engine.service linked disabled\nflatshare.target linked enabled\n",
    )
    (box / ".gpu-apps").write_text("")
    write(box / ".listen-tcp", "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n")
    write(box / ".listen-udp", "")
    write(
        hosts / "cyan" / "flatshare" / "data" / "callers" / "manifest.json",
        json.dumps({"hello": {"kind": "turn"}}),
    )
    write(hosts / "cyan" / "flatshare" / "data" / "callers" / "hello.wav", "RIFF")
    write(hosts / "peacock" / "flatshare-data" / "weka-x32" / "t.json", "{}")
    # The workstation's corpus, 50 traces, and the checkout to push: the kit's push.sh and pull.sh, and stubs.
    corpus = tmp_path / "corpus" / "weka-x32"
    corpus.mkdir(parents=True)
    for n in range(50):
        (corpus / f"trace-{n:02d}.json").write_text(
            json.dumps({"requests": [{"in": 100 * (n + 1)}]})
        )
    repo = tmp_path / "checkout"
    for rel in (
        "deploy/a100/push.sh",
        "deploy/a100/pull.sh",
        "deploy/phastform/engine.yaml",
        "deploy/phastform/stt.yaml",
        "benchmarks/e2/long_traces.py",
        "deploy/bin/in-private-net",
    ):
        write(repo / rel, (REPO / rel).read_text(), 0o755)
    write(repo / "deploy" / "a100" / "bootstrap.sh", STUB_BOOTSTRAP, 0o755)
    write(repo / "benchmarks" / "a100" / "run.sh", STUB_RUN, 0o755)
    write(repo / "deploy" / "bin" / "shims" / "systemctl", STUB_SYSTEMCTL, 0o755)
    write(repo / "deploy" / "bin" / "shims" / "journalctl", STUB_JOURNALCTL, 0o755)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "the kit"], check=True)
    bin_dir = tmp_path / "bin"
    for name, text in TOOLS.items():
        write(bin_dir / name, text, 0o755)
    # A host whose kernel refuses unprivileged user namespaces.
    write(tmp_path / "no-userns" / "unshare", "#!/bin/sh\nexit 1\n", 0o755)
    (bin_dir / "ssh").symlink_to(FAKES / "campaign" / "ssh")
    work = tmp_path / "work"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("REHEARSAL_", "FLATSHARE_", "XDG_", "HF_")) and k != "VIRTUAL_ENV"
    } | {
        "PATH": os.pathsep.join([str(bin_dir), os.environ["PATH"]]),
        "FAKE_HOSTS": str(hosts),
        "FAKE_BOX": str(box),
        # The login environment's session bus, through which the box's user manager answers.
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={box}/bus",
        "REHEARSAL_HOST": "phastform",
        "REHEARSAL_SSH": "ssh",
        "REHEARSAL_CYAN": "cyan",
        "REHEARSAL_PEACOCK": "peacock",
        "REHEARSAL_REPO": str(repo),
        "REHEARSAL_WORK": str(work),
        "REHEARSAL_CORPUS": str(corpus),
        "REHEARSAL_MIN_FREE_GIB": "1",
        "REHEARSAL_POLL_S": "0.2",
    }
    return {
        "tmp": tmp_path,
        "hosts": hosts,
        "box": box,
        "hub": hub,
        "repo": repo,
        "env": env,
        "work": work,
        "state": work / "state",
    }


def rehearse(world: dict, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REHEARSE), *args],
        env=world["env"] | env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def wait_for(path: Path, timeout: float = 10.0) -> Path:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)
    return path


def gone(pid: int, timeout: float = 10.0) -> bool:
    """Whether ``pid`` exits within ``timeout``: a killed orphan lasts until init reaps it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def box_snapshot(box: Path) -> dict[str, str]:
    """The box's deployment: each file's contents and each link's target, under the trees the rehearsal must leave."""
    found = {}
    for top in ("flatshare", ".config", ".cache"):
        for path in sorted((box / top).rglob("*")):
            if path.is_symlink():
                found[str(path.relative_to(box))] = "-> " + os.readlink(path)
            elif path.is_file():
                found[str(path.relative_to(box))] = path.read_text(errors="replace")
    return found


def test_the_checks_refuse_a_box_that_runs_a_busy_gpu_and_a_held_port(world: dict) -> None:
    box = world["box"]
    write(box / ".units", "flatshare-engine.service loaded active running Flatshare LLM engine\n")
    write(box / ".gpu-apps", "4242, VLLM::EngineCore, 13000 MiB\n")
    write(box / ".listen-tcp", "LISTEN 0 128 127.0.0.1:18080 0.0.0.0:*\n")
    write(box / ".listen-udp", "UNCONN 0 0 0.0.0.0:17882 0.0.0.0:*\n")
    done = rehearse(world, "check")
    assert done.returncode == 1, done.stdout + done.stderr
    out = done.stdout
    assert "check: the box's units: refused: running: flatshare-engine.service" in out
    assert "check: GPU: refused: in use: 4242, VLLM::EngineCore" in out
    assert "check: ports: refused: held: tcp 127.0.0.1:18080 udp 0.0.0.0:17882" in out
    assert not (world["state"] / "check.done").exists()
    # Each cleared, the host is ready; nothing was written on it.
    write(box / ".units", "flatshare-engine.service loaded inactive dead Flatshare LLM engine\n")
    (box / ".gpu-apps").write_text("")
    write(box / ".listen-tcp", "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n")
    (box / ".listen-udp").write_text("")
    done = rehearse(world, "check")
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"check: hub cache: ok: {world['hub']}" in done.stdout
    assert f"check: model {QWEN}: ok: {QWEN_REV}" in done.stdout
    assert not (box / "a100-rehearsal").exists()


def test_the_rehearsal_runs_every_step_in_its_own_home_and_leaves_the_box_as_it_was(
    world: dict,
) -> None:
    box, hosts, state = world["box"], world["hosts"], world["state"]
    before = box_snapshot(box)
    for step in ("check", "init", "data", "push", "stage", "gpu", "warm"):
        done = rehearse(world, step)
        assert done.returncode == 0, (step, done.stdout + done.stderr)
    home = box / "a100-rehearsal"
    root = home / "flatshare"
    # The rehearsal's environment: its own home, runtime directory and state, flatshare-svc, the box's hub cache
    # offline, no session bus, sudo refused; and the box's user manager out of its reach.
    recorded = (root / "bootstrap-stage.env").read_text().splitlines()
    assert f"HOME={home}" in recorded and f"XDG_RUNTIME_DIR={home}/run" in recorded
    assert f"HF_HUB_CACHE={world['hub']}" in recorded and "HF_HUB_OFFLINE=1" in recorded
    assert "FLATSHARE_SUPERVISOR=flatshare-svc" in recorded and "sudo refused" in recorded
    assert not any(line.startswith("DBUS_SESSION_BUS_ADDRESS=") for line in recorded)
    assert (
        "the box's user manager is out of the rehearsal's reach"
        in (state / "logs" / "init.log").read_text()
    )
    assert (home / "run").stat().st_mode & 0o777 == 0o700
    # The box's Kokoro, copied; the data pushed as files; the source at the checkout's revision, the box's untouched.
    assert (
        root / "vendor" / "Kokoro-FastAPI" / "api" / "src" / "models" / "v1_0" / "kokoro-v1_0.pth"
    ).exists()
    long = sorted((root / "data" / "weka-x32-long").iterdir())
    assert len(long) == 48 and not any(p.is_symlink() for p in long)
    assert (root / "data" / "callers" / "manifest.json").exists()
    revision = subprocess.run(
        ["git", "-C", str(world["repo"]), "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert (root / "src" / "DEPLOYED").read_text().strip() == revision
    assert box_snapshot(box) == before
    # push holds the host's data to FLATSHARE_DATA's: a file there that FLATSHARE_DATA lacks (push.sh's rsync deletes
    # none) fails it.
    stale = root / "data" / "weka-x32-long" / "stale.json"
    stale.write_text("{}")
    done = rehearse(world, "--redo", "push", "push")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "push: the host's data is not FLATSHARE_DATA's" in done.stdout
    stale.unlink()
    # The control plane up under the rehearsal's flatshare-svc; the box's user manager was asked only for what it runs.
    calls = (box / ".systemctl.log").read_text().splitlines()
    assert calls and all("list-unit" in c for c in calls), calls
    assert (
        "start flatshare-voice-worker.service"
        in (home / ".local" / "state" / "fake-flatshare-svc" / "calls").read_text()
    )
    # A step done is not run again; its host step is done.
    done = rehearse(world, "stage")
    assert done.returncode == 0 and "stage: done before" in done.stdout, done.stdout + done.stderr
    assert "bootstrap stage" not in done.stdout
    # Block 0: a listener audit that refuses stops the rehearsal; one that passes with the GPU servers in the host's
    # namespace, where this host allows user namespaces, fails block 0's check; a step done but not checked runs
    # again only under --redo (the block's blocks/0.done removed); with each GPU server in a namespace of its own
    # (deploy/bin/in-private-net), block 0 is done.
    done = rehearse(world, "block0", FAKE_AUDIT="refused")
    assert done.returncode == 1 and "block0: run.sh 0 failed" in done.stdout, (
        done.stdout + done.stderr
    )
    assert not (state / "block0.done").exists()
    done = rehearse(world, "block0", FAKE_AUDIT="degraded")
    assert done.returncode == 1, done.stdout + done.stderr
    assert (
        "the listener audit's line is not ok with flatshare-engine, flatshare-stt, flatshare-tts in network "
        "namespaces of their own" in done.stdout
    )
    done = rehearse(world, "block0")
    assert done.returncode == 1 and "block0 is done" in done.stdout, done.stdout
    done = rehearse(world, "--redo", "block0", "block0")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "in network namespaces of their own" in done.stdout
    # Where the kernel refuses user namespaces (the A100's Ubuntu 24.04), the degraded line is the one expected.
    done = subprocess.run(
        ["bash", str(HOST_SH), "verify", "block0"],
        cwd=home,
        env=world["env"]
        | {"HOME": str(home), "PATH": f"{world['tmp'] / 'no-userns'}:{world['env']['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 1 and "FLATSHARE_PRIVATE_NET=host" in done.stdout, done.stdout
    # Block 1, then block 4 for its first point and the deadline's stop, its deadline 16 to 17 minutes on.
    for step in ("block1", "block4"):
        done = rehearse(world, step)
        assert done.returncode == 0, (step, done.stdout + done.stderr)
    args = (root / "runs" / "a100" / "block4.args").read_text().split()
    assert args[0] == "--until" and args[2:] == ["--estimate", "4=15", "4"], args
    hours, minutes = map(int, args[1].split(":"))
    now = time.gmtime()
    ahead = (hours * 60 + minutes - now.tm_hour * 60 - now.tm_min) % 1440
    assert 16 <= ahead <= 17, (args[1], ahead)
    assert "the first point complete, then: w5-a1-voice: 9 minutes would end after" in done.stdout
    # pull copies the rehearsal's records, not the box's; reproduce runs make on peacock over them.
    done = rehearse(world, "pull")
    assert done.returncode == 0, done.stdout + done.stderr
    pulled = world["work"] / "hosts" / "phastform"
    assert (pulled / "runs" / "a100" / "blocks.log").exists()
    assert not (pulled / "runs" / "box-only.txt").exists()
    assert "a line of the engine's" in (pulled / "runs" / "journal" / "units.log").read_text()
    done = rehearse(world, "reproduce")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (world["work"] / "reproduce" / "a100.md").read_text().startswith("# The A100 40 GB run")
    assert "make reproduce-a100 RECORDS=../records" in (hosts / "peacock" / "make.log").read_text()
    # cleanup: the rehearsal's units stopped, the box as it was; one change to the box is found and named.
    done = rehearse(world, "cleanup")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verify: the box's deployment as the rehearsal found it" in done.stdout
    stops = [
        line
        for line in (home / ".local" / "state" / "fake-flatshare-svc" / "calls")
        .read_text()
        .splitlines()
        if line.startswith("systemctl --user stop ")
    ]
    assert stops and all(
        unit in stops[-1].split()
        for unit in (
            "flatshare.target",
            "flatshare-engine.service",
            "flatshare-voice-worker.service",
            "flatshare-gateway.socket",
        )
    ), stops
    assert box_snapshot(box) == before
    # A unit that its stop leaves deactivating is not stopped: cleanup fails, naming it.
    done = rehearse(world, "--redo", "cleanup", "cleanup", FAKE_UNSTOPPABLE="flatshare-engine")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "stop-units: not stopped: flatshare-engine.service (deactivating)" in done.stdout
    deployed = box / "flatshare" / "src" / "DEPLOYED"
    found = deployed.stat()
    deployed.write_text("changed\n")
    done = rehearse(world, "--redo", "cleanup", "cleanup")
    assert done.returncode == 1, done.stdout + done.stderr
    assert (
        "verify: the box's deployment changed" in done.stdout
        and "flatshare/src/DEPLOYED" in done.stdout
    )
    status = rehearse(world, "status")
    assert status.returncode == 0 and "block4     done" in status.stdout, status.stdout
    # The box as found again (the file's contents and mtime, which the state records): --purge runs cleanup again
    # although it is done, and removes the rehearsal's home.
    deployed.write_text("v0.3-191-g987fd09\n")
    os.utime(deployed, ns=(found.st_atime_ns, found.st_mtime_ns))
    done = rehearse(world, "--purge", "cleanup")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "cleanup: done: the box's deployment as found; ~/a100-rehearsal removed" in done.stdout
    assert not home.exists() and box_snapshot(box) == before


def test_a_failed_command_ends_its_step_and_init_wants_its_probe_answered(world: dict) -> None:
    # Each step runs with set -e in force: a command that fails ends the step, with its status. Here the connection to
    # the host drops at init's line of the rehearsal's environment, so init ends there with ssh's 255, its probe unasked.
    hosts, state = world["hosts"], world["state"]
    assert rehearse(world, "check").returncode == 0
    done = rehearse(world, "init", FAKE_SSH_DOWN="phastform:the rehearsal environment")
    assert done.returncode == 255, done.stdout + done.stderr
    assert "init: failed, exit status 255" in done.stdout
    assert "is-system-running" not in (hosts / "ssh.log").read_text()
    # The isolation probe passes only on an answer that the rehearsal's systemctl --user reaches no manager: one that
    # cannot be had, the connection dropped at it, fails init, and so does one that says something else.
    (hosts / "phastform.down").unlink()
    done = rehearse(world, "init", FAKE_SSH_DOWN="phastform:is-system-running")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "init: the rehearsal's environment did not answer" in done.stdout
    (hosts / "phastform.down").unlink()
    done = rehearse(world, "init", FAKE_NO_BUS="unknown")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "may reach the box's user manager (systemctl --user there: unknown)" in done.stdout
    assert not (state / "init.done").exists()
    done = rehearse(world, "init")
    assert done.returncode == 0 and (state / "init.done").exists(), done.stdout + done.stderr
    # cleanup likewise fails, rather than finding nothing to clean, where the host does not answer whether it has the
    # rehearsal's home.
    done = rehearse(world, "cleanup", FAKE_SSH_DOWN="phastform:rehearsal.env")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "cleanup: phastform did not answer whether it has ~/a100-rehearsal" in done.stdout
    assert not (state / "cleanup.done").exists()


def test_reproduce_copies_this_workstations_corpus_to_a_peacock_that_has_none(world: dict) -> None:
    # Where peacock has no weka-x32 at REHEARSAL_PEACOCK_CORPUS, reproduce copies this workstation's into its own
    # directory there and runs make over that. pull done, with the records it pulled, is what reproduce asks for.
    hosts, work = world["hosts"], world["work"]
    write(work / "hosts" / "phastform" / "runs" / "a100" / "blocks.log", "run: blocks 0 1 4\n")
    write(world["state"] / "pull.done", "pulled\n")
    (work / "data").mkdir()
    (work / "data" / "weka-x32").symlink_to(world["tmp"] / "corpus" / "weka-x32")
    done = rehearse(world, "reproduce", REHEARSAL_PEACOCK_CORPUS="flatshare-data/none")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        "no flatshare-data/none on peacock: copying this workstation's weka-x32 there"
        in done.stdout
    )
    copied = hosts / "peacock" / "flatshare-rehearsal-reproduce" / "corpus" / "weka-x32"
    assert len(list(copied.glob("trace-*.json"))) == 50
    assert f"CORPUS={copied}" in (hosts / "peacock" / "make.log").read_text()


@PROC
def test_cleanup_ends_the_rehearsals_processes_whatever_their_titles(world: dict) -> None:
    # AIPerf's and vLLM's processes set their titles, which blanks the HOME=~/a100-rehearsal that leftovers finds the
    # rehearsal's processes by. AIPerf runs in a session of its own, so a step killed before flatshare-bench stopped it
    # leaves it running: cleanup ends the process groups flatshare-bench names for its tools, by flatshare-bench's own
    # kill_left. A unit's engine core is in its server's group: leftovers ends the groups and sessions of the processes
    # it finds.
    pytest.importorskip("setproctitle")
    from flatshare_bench.e2.tools import Process

    for step in ("check", "init"):
        assert rehearse(world, step).returncode == 0
    tmp, home = world["tmp"], world["box"] / "a100-rehearsal"
    # The rehearsal's bench venv: this test's Python, which has flatshare-bench.
    bench = home / "flatshare" / "venvs" / "bench" / "bin" / "python"
    write(bench, f'#!/bin/sh\nexec {sys.executable} "$@"\n', 0o755)
    (tmp / "titled.py").write_text(TITLED)
    env = os.environ | {"HOME": str(home)}
    titled = [sys.executable, str(tmp / "titled.py")]
    # AIPerf as flatshare-bench starts it, its group named in the tools' directory, and the run then killed.
    command = [*titled, "aiperf system_controller", str(tmp / "aiperf.pids")]
    aiperf = Process("AIPerf", command, tmp / "aiperf.log", env, str(home / "flatshare"))
    # A unit's server in a session of its own, with its engine core, and a process whose environment names another
    # home: not the rehearsal's, whatever its session.
    command = [*titled, "", str(tmp / "server.pids"), str(tmp)]
    server = subprocess.Popen(command, env=env, start_new_session=True)
    (worker,) = map(int, wait_for(tmp / "aiperf.pids").read_text().split())
    core, other = map(int, wait_for(tmp / "server.pids").read_text().split())
    try:
        for pid in (aiperf.popen.pid, worker, core):
            deadline = time.monotonic() + 10
            while b"=" in Path(f"/proc/{pid}/environ").read_bytes():
                assert time.monotonic() < deadline, f"{pid} set no title"
                time.sleep(0.05)
        done = rehearse(world, "cleanup")
        assert done.returncode == 0, done.stdout + done.stderr
        killed = " ".join(map(str, sorted([aiperf.popen.pid, worker])))
        assert f"kill-tools: flatshare-bench's kill_left ended {killed}\n" in done.stdout, (
            done.stdout
        )
        assert aiperf.popen.wait(timeout=10) == -signal.SIGKILL
        assert server.wait(timeout=10) == -signal.SIGTERM
        assert gone(worker) and gone(core), done.stdout
        assert "leftovers: none left after TERM and KILL" in done.stdout
        assert Path(f"/proc/{other}/stat").read_text().rsplit(") ", 1)[1][0] != "Z"
    finally:
        for pid in (aiperf.popen.pid, worker, server.pid, core, other):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        aiperf.popen.wait()
        server.wait()


def test_a_step_on_the_host_outlives_its_session_and_is_waited_for_once(tmp_path: Path) -> None:
    # host.sh's detached steps, on this host: a step started runs on after its starter returns, a second start while
    # it runs starts nothing, wait returns its exit status, and stop ends one that runs.
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ | {"HOME": str(home), "REHEARSAL_POLL_S": "0.1"}

    def host_sh(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(HOST_SH), *args], env=env, capture_output=True, text=True, timeout=60
        )

    started = host_sh("start", "slow", "sleep 1; echo slept; exit 3")
    assert started.returncode == 0 and "slow started (running" in started.stdout, started.stdout
    assert host_sh("state", "slow").stdout.startswith("running ")
    again = host_sh("start", "slow", "echo twice")
    assert "runs already" in again.stdout
    waited = host_sh("wait", "slow")
    assert waited.returncode == 3 and "slept" in waited.stdout, waited.stdout
    assert host_sh("state", "slow").stdout.strip() == "done 3"
    assert "twice" not in (home / ".rehearsal" / "steps" / "slow.log").read_text()
    host_sh("start", "long", "sleep 30")
    stopped = host_sh("stop", "long")
    assert (
        "long stopped" in stopped.stdout and host_sh("state", "long").stdout.strip() == "done 143"
    )
    done = host_sh("start", "ok", "true")
    assert done.returncode == 0
    assert host_sh("wait", "ok").returncode == 0
    assert "ok is done" in host_sh("start", "ok", "false").stdout


def test_a_stopped_step_is_given_the_time_its_tools_take_to_stop(tmp_path: Path) -> None:
    # stop's TERM reaches the step's process group, and stop waits for all of the group, not its leader alone: the
    # step's bash ends at once, while flatshare-bench, on TERM, stops its callers and AIPerf before it exits; here 2 s.
    home = tmp_path / "home"
    home.mkdir()
    flag = tmp_path / "point"
    (tmp_path / "member.py").write_text(
        "import signal, sys, time\n"
        "def stop(*_):\n"
        "    time.sleep(2)\n"
        "    open(sys.argv[1] + '.stopped', 'w').close()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "open(sys.argv[1] + '.ready', 'w').close()\n"
        "time.sleep(60)\n"
    )
    env = os.environ | {"HOME": str(home)}

    def host_sh(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(HOST_SH), *args], env=env, capture_output=True, text=True, timeout=60
        )

    command = shlex.join([sys.executable, str(tmp_path / "member.py"), str(flag)])
    assert host_sh("start", "point", command).returncode == 0
    wait_for(Path(f"{flag}.ready"))
    stopped = host_sh("stop", "point")
    assert "point stopped" in stopped.stdout, stopped.stdout + stopped.stderr
    assert Path(f"{flag}.stopped").exists(), "the step's group was killed before its member stopped"
    assert host_sh("state", "point").stdout.strip() == "done 143"


def host_state(home: Path, name: str, **env: str) -> str:
    """host.sh state NAME, with HOME and any of the environment given."""
    done = subprocess.run(
        ["bash", str(HOST_SH), "state", name],
        env=os.environ | {"HOME": str(home)} | env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.strip()


@PROC
def test_a_running_step_reads_as_running_whatever_its_environment_holds(tmp_path: Path) -> None:
    # host.sh tells a step's process by FLATSHARE_REHEARSAL_STEP in its /proc/PID/environ. Read through tr and grep -q
    # under pipefail, tr died of SIGPIPE once grep had matched while more of the environment was to come, and the
    # running step read as died: here deterministically, the environment after the mark being larger than a pipe's
    # buffer.
    steps = tmp_path / ".rehearsal" / "steps"
    steps.mkdir(parents=True)
    rest = {k: v for k, v in os.environ.items() if k != "FLATSHARE_REHEARSAL_STEP"}
    env = {
        "FLATSHARE_REHEARSAL_STEP": "stage",
        **rest,
        "FLATSHARE_BULK_A": "y" * 100_000,
        "FLATSHARE_BULK_B": "y" * 100_000,
    }
    step = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env)
    try:
        (steps / "stage.pid").write_text(f"{step.pid}\n")
        assert host_state(tmp_path, "stage") == f"running {step.pid}"
    finally:
        step.kill()
        step.wait()


def test_a_steps_status_once_written_is_final_whatever_its_pid_names(tmp_path: Path) -> None:
    # Once its status is written, a step reads done although its pid names a live process, as one reused since can:
    # here this test's own, which where no /proc tells a process's mark (macOS) would read as running.
    steps = tmp_path / ".rehearsal" / "steps"
    steps.mkdir(parents=True)
    (steps / "stage.pid").write_text(f"{os.getpid()}\n")
    (steps / "stage.status").write_text("0\n")
    assert host_state(tmp_path, "stage") == "done 0"


@PROC
def test_a_step_that_ends_while_its_process_is_looked_at_reads_done(tmp_path: Path) -> None:
    # state looks at the status, then the process, then the status again: a step writes its status before it exits,
    # so one seen gone has written any status it was to write. Here the step ends during the look at its process: a
    # grep that reads its environment writes its status, and answers that the mark is gone.
    steps = tmp_path / ".rehearsal" / "steps"
    steps.mkdir(parents=True)
    status = shlex.quote(str(steps / "stage.status"))
    write(tmp_path / "bin" / "grep", f"#!/bin/sh\necho 0 >{status}\nexit 1\n", 0o755)
    env = os.environ | {"FLATSHARE_REHEARSAL_STEP": "stage"}
    step = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env)
    try:
        (steps / "stage.pid").write_text(f"{step.pid}\n")
        path = f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"
        assert host_state(tmp_path, "stage", PATH=path) == "done 0"
    finally:
        step.kill()
        step.wait()


@PROC
def test_a_pid_file_naming_another_live_process_reads_as_no_running_step(tmp_path: Path) -> None:
    # A process is the step's only while its environment carries the step's mark, and a status once written is final.
    # A pid file that names another live process of this user reads died, whether its environment lacks the mark or
    # cannot be read (a non-dumpable process, as ssh-agent is: prctl PR_SET_DUMPABLE 0), and done once the step's
    # status is written.
    steps = tmp_path / ".rehearsal" / "steps"
    steps.mkdir(parents=True)
    rest = {k: v for k, v in os.environ.items() if k != "FLATSHARE_REHEARSAL_STEP"}
    sleep = "import time; time.sleep(60)"
    hidden = "import ctypes; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); " + sleep
    plain = subprocess.Popen([sys.executable, "-c", sleep], env=rest)
    marked = rest | {"FLATSHARE_REHEARSAL_STEP": "stage"}
    agent = subprocess.Popen([sys.executable, "-c", hidden], env=marked)
    try:
        deadline = time.monotonic() + 10
        while os.access(f"/proc/{agent.pid}/environ", os.R_OK):
            assert time.monotonic() < deadline, "the process stayed dumpable"
            time.sleep(0.05)
        for process in (plain, agent):
            (steps / "stage.pid").write_text(f"{process.pid}\n")
            (steps / "stage.status").unlink(missing_ok=True)
            assert host_state(tmp_path, "stage") == f"died {process.pid}"
            (steps / "stage.status").write_text("0\n")
            assert host_state(tmp_path, "stage") == "done 0"
    finally:
        for process in (plain, agent):
            process.kill()
            process.wait()


def test_wait_follows_a_log_that_grows_as_it_is_read(tmp_path: Path) -> None:
    # wait shows the step's log up to the size it saw. A log that grows past that size before it is read must not end
    # wait: read with tail in front of head, under pipefail, tail died of SIGPIPE once head had its bytes, and wait
    # exited 141, which rehearse.sh takes for the step's status. Here wc grows the log by 1 MiB as it answers.
    home = tmp_path / "home"
    home.mkdir()
    log = shlex.quote(str(home / ".rehearsal" / "steps" / "burst.log"))
    grow = f'#!/usr/bin/env bash\ncommand -p wc "$@"\nstatus=$?\nhead -c 1048576 /dev/zero >>{log}\nexit $status\n'
    write(tmp_path / "bin" / "wc", grow, 0o755)

    def host_sh(*args: str, **more: str) -> subprocess.CompletedProcess[str]:
        env = os.environ | {"HOME": str(home), "REHEARSAL_POLL_S": "0.1"} | more
        return subprocess.run(
            ["bash", str(HOST_SH), *args], env=env, capture_output=True, text=True, timeout=60
        )

    assert host_sh("start", "burst", "echo a burst of output; exit 7").returncode == 0
    deadline = time.monotonic() + 10
    while host_state(home, "burst") != "done 7":
        assert time.monotonic() < deadline, host_state(home, "burst")
        time.sleep(0.05)
    waited = host_sh("wait", "burst", PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    assert waited.returncode == 7, waited.stdout + waited.stderr
    assert "a burst of output" in waited.stdout and "burst ended, exit status 7" in waited.stdout


def test_a_step_that_ends_before_its_pid_file_is_written_does_not_start(tmp_path: Path) -> None:
    # start returns once the step's bash has written its pid file, and fails, its command not run, when that bash
    # ends first: here the pid file's temporary name is a directory. It waited 5 s for the pid file, then returned 0
    # with the command run unwatched.
    home = tmp_path / "home"
    (home / ".rehearsal" / "steps" / "lost.pid.new").mkdir(parents=True)
    ran = tmp_path / "ran"
    started = subprocess.run(
        ["bash", str(HOST_SH), "start", "lost", f"touch {shlex.quote(str(ran))}"],
        env=os.environ | {"HOME": str(home)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert started.returncode == 1, started.stdout + started.stderr
    assert "rehearsal: lost did not start" in started.stdout
    time.sleep(0.5)
    assert not ran.exists() and host_state(home, "lost") == "none"


def test_rehearsal_ssh_names_the_home_it_needs_in_full() -> None:
    # Its message, an unquoted word in ${REHEARSAL_HOME:?...}, lost both apostrophes, which bash took for quotes:
    # "names the rehearsals home under the hosts login home".
    env = {name: value for name, value in os.environ.items() if name != "REHEARSAL_HOME"}
    done = subprocess.run(
        ["bash", str(HERE / "bin" / "rehearsal-ssh"), "host", "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode != 0
    assert "REHEARSAL_HOME names the rehearsal's home under the host's login home" in done.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
