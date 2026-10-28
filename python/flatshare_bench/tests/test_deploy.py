"""The deployment: what the GPU nodes serve and the tunnels reach, cyan's install, and the script that
authorizes cyan's tunnel keys on a GPU node, run against stand-ins for ssh and systemd."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import tomllib
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
from servers import UnixServer, serve, socket_file, stop

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
KEYS = {
    "engine": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMa9oQ1uNw4Z5fT1n2x0bq7d1H0mGJ1s7mYcQjv8Yw5k",
    "stt": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK3fW8pQ2nV7xZ1cR5tY9uB4mE6hJ0aLsD2gF8kN3qPw",
    "tts": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHq2wE4rT6yU8iO0pA1sD3fG5hJ7kL9zX2cV4bN6mQ8w",
}
"""cyan's tunnel keys, by the socket each one reaches: their type and blob, as ssh-keygen writes them before the
comment."""
FORMER = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPz4xY6wQ8eR0tU2iO4pA6sD8fG0hJ2kL4zX6cV8bN0mQ2e"
"""cyan's former tunnel key, which forwarded the node's ports."""
FORMER_ENGINE = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL5nT7wQ9eS1uV3iP5pB7sF9gH1jK3lZ5xC7vB9nM1qR"
"""cyan's engine key under the project's former name, whose entry reached $XDG_RUNTIME_DIR/lh."""
LOGIN = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB7m2Qp9xR4tYc3vLk8wNs6fJd0aHe5uPq1zXo2iGb4M me@laptop"
"""Another key in a GPU node's authorized_keys: its user's own."""

FAKE_SSH = """\
#!/bin/sh
# ssh HOST COMMAND...: runs COMMAND in HOST's home under $FAKE_HOSTS, as sshd runs it in the user's shell.
host=$1
shift
cd "$FAKE_HOSTS/$host" && exec sh -c "$*"
"""
OPENBSD_NC = """\
#!/bin/sh
# nc -h on a GPU node: OpenBSD's netcat, as Ubuntu's netcat-openbsd prints its usage.
echo "OpenBSD netcat (Debian patchlevel 1.218-4ubuntu1)" >&2
exit 1
"""


def authorize(
    work: Path,
    keys: dict[str, str],
    authorized_keys: str = "",
    formers: dict[str, str] | None = None,
    nc: str = OPENBSD_NC,
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Runs authorize-tunnel.sh from cyan to phastform, hosts kept under ``work``: cyan's tunnel keys are
    ``keys`` by socket (and its former keys ``formers``, by file name), phastform's authorized_keys ``authorized_keys``
    and its nc ``nc``. Returns the run and the file it leaves."""
    hosts, bin_dir = work / "hosts", work / "bin"
    for host in ("cyan", "phastform"):
        (hosts / host / ".ssh").mkdir(parents=True)
    for name, key in keys.items():
        (hosts / "cyan" / ".ssh" / f"flatshare_tunnel_{name}_ed25519.pub").write_text(f"{key}\n")
    for stem, key in (formers or {}).items():
        (hosts / "cyan" / ".ssh" / f"{stem}_ed25519.pub").write_text(f"{key}\n")
    authorized = hosts / "phastform" / ".ssh" / "authorized_keys"
    authorized.write_text(authorized_keys)
    bin_dir.mkdir()
    for tool, text in (("fake-ssh", FAKE_SSH), ("nc", nc)):
        (bin_dir / tool).write_text(text)
        (bin_dir / tool).chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_HOSTS": str(hosts),
        "FLATSHARE_SSH": "fake-ssh",
    }
    done = subprocess.run(
        ["bash", str(DEPLOY / "authorize-tunnel.sh"), "cyan", "phastform"],
        env=env,
        capture_output=True,
        text=True,
    )
    return done, authorized.read_text()


def test_the_gpu_nodes_publish_no_kv_events_and_cyan_reaches_them_only_through_the_tunnel() -> None:
    # vLLM's KV-event publisher connects to a TCP endpoint and serves a replay of the events, which carry
    # every cached block's token ids and session id: on a shared host any user could bind or ask for them,
    # and nothing of the deployment reads them.
    engine = (DEPLOY / "phastform" / "engine.yaml").read_text()
    assert "kv-events-config" not in engine and "tcp://" not in engine
    cyan = DEPLOY / "cyan" / "systemd"
    tunnel = (cyan / "flatshare-tunnel@.service").read_text()
    assert exec_lines(tunnel, "ExecStart") == [
        "/usr/bin/python3 %h/flatshare/src/deploy/bin/flatshare-tunnel %i"
    ]
    assert "EnvironmentFile=%h/.config/flatshare/tunnel-%i.env" in tunnel.splitlines()
    # One template for every GPU node: no unit forwards a port, or names a tunnel unit that is gone.
    for unit in cyan.iterdir():
        text = unit.read_text()
        assert "ssh " not in text and "flatshare-tunnel-" not in text, unit.name


def test_the_tunnel_unit_does_not_restart_a_configuration_it_refuses() -> None:
    # flatshare-tunnel exits 2 only for a configuration it refuses (no target, a session count under 1), which a
    # restart would refuse again every RestartSec; any other exit is restarted.
    tunnel = (DEPLOY / "cyan" / "systemd" / "flatshare-tunnel@.service").read_text().splitlines()
    assert "Restart=always" in tunnel and "RestartPreventExitStatus=2" in tunnel


def test_cyans_servers_and_tunnels_run_on_their_planes_cores() -> None:
    # The control plane has cores 0-7 and the voice plane 32-47, the load generators the rest
    # (e2-host.toml): a unit on every core competes with what a point measures. The tunnel's SSH
    # sessions carry every byte to and from the GPU nodes, and ran on no core set (RA's W6 review).
    cores = {
        "flatshare-gateway.service": "0-7",
        "flatshare-smg.service": "0-7",
        "flatshare-tunnel@.service": "0-7",
        "flatshare-livekit.service": "32-47",
        "flatshare-voice-worker.service": "32-47",
    }
    cyan = DEPLOY / "cyan" / "systemd"
    for name, wanted in cores.items():
        assert exec_lines((cyan / name).read_text(), "CPUAffinity") == [wanted], name
    # Every service of cyan's is one of them, or the voice plane's namespace holder, which only waits.
    assert {p.name for p in cyan.glob("*.service")} == {*cores, "flatshare-voicenet.service"}


def test_each_tunnel_key_may_only_bridge_its_session_to_its_socket(tmp_path: Path) -> None:
    socket_dir = f"/run/user/{os.getuid()}/flatshare"
    earlier = f'restrict,port-forwarding,permitopen="127.0.0.1:5558" {KEYS["stt"]} flatshare-tunnel-stt@cyan'
    former = f'restrict,port-forwarding,permitopen="127.0.0.1:8000",command="/bin/false" {FORMER} flatshare-tunnel@cyan'
    # The keys as ssh-keygen writes them, and without their comments, which are optional.
    for case, comment in (("commented", " flatshare-tunnel-{}@cyan"), ("bare", "")):
        keys = {name: key + comment.format(name) for name, key in KEYS.items()}
        done, authorized = authorize(
            tmp_path / case,
            keys,
            f"{LOGIN}\n{earlier}\n{former}\n",
            formers={"flatshare_tunnel": FORMER},
        )
        assert done.returncode == 0, done.stderr
        # The earlier entry for a key and the former key's go; every other key stays. Each entry allows its
        # key's sessions one command, the bridge to the node's socket of its name, and no forwarding.
        assert authorized.splitlines() == [LOGIN] + [
            f'restrict,command="exec nc -q0 -U {socket_dir}/{name}.sock" {keys[name]}'
            for name in KEYS
        ], case
        assert done.stdout == f"authorized: engine, stt and tts on {socket_dir} (3 entries)\n", case


def test_the_former_names_keys_lose_their_entries_too(tmp_path: Path) -> None:
    # A host not moved since the project's rename keeps its keys under the former name: the entries of the key that
    # forwarded the node's ports and of the sockets' keys, which reach $XDG_RUNTIME_DIR/lh, go as the former key's does.
    uid = os.getuid()
    entries = [
        f'restrict,port-forwarding,permitopen="127.0.0.1:8000",command="/bin/false" {FORMER} lh-tunnel@cyan',
        f'restrict,command="exec nc -q0 -U /run/user/{uid}/lh/engine.sock" {FORMER_ENGINE} lh-tunnel-engine@cyan',
    ]
    formers = {"lh_tunnel": FORMER, "lh_tunnel_engine": FORMER_ENGINE}
    done, authorized = authorize(tmp_path, KEYS, "\n".join([LOGIN, *entries, ""]), formers=formers)
    assert done.returncode == 0, done.stderr
    assert authorized.splitlines() == [LOGIN] + [
        f'restrict,command="exec nc -q0 -U /run/user/{uid}/flatshare/{name}.sock" {KEYS[name]}'
        for name in KEYS
    ]


def test_authorizing_needs_all_three_keys_and_openbsd_netcat(tmp_path: Path) -> None:
    # A client without one of the keys changes nothing.
    done, authorized = authorize(tmp_path / "missing", {"engine": KEYS["engine"]}, f"{LOGIN}\n")
    assert (
        done.returncode == 2
        and "no public key in cyan's .ssh/flatshare_tunnel_stt_ed25519.pub" in done.stderr
    )
    assert authorized == f"{LOGIN}\n"
    # Nor does a node whose nc cannot bridge a session to a socket.
    gnu = "#!/bin/sh\necho 'GNU netcat 0.7.1, a rewrite of the famous networking tool.'\n"
    done, authorized = authorize(tmp_path / "gnu", KEYS, f"{LOGIN}\n", nc=gnu)
    assert done.returncode == 2 and "is not OpenBSD netcat" in done.stderr
    assert authorized == f"{LOGIN}\n"


def authorize_guard(
    work: Path, authorized_keys: str = "", guard: bool = True, key: str | None = None
) -> tuple[subprocess.CompletedProcess[str], str, Path]:
    """Runs authorize-guard.sh from cyan to phastform, hosts kept under ``work``: cyan with its guard key ``key`` (none
    when None: the script makes one), phastform with ``authorized_keys`` and, with ``guard``, the deployment's
    node-report. Returns the run, the node's authorized_keys after it and cyan's home."""
    hosts, bin_dir = work / "hosts", work / "bin"
    for host in ("cyan", "phastform"):
        (hosts / host / ".ssh").mkdir(parents=True)
    if key is not None:
        (hosts / "cyan" / ".ssh" / "flatshare_guard_ed25519").write_text("private\n")
        (hosts / "cyan" / ".ssh" / "flatshare_guard_ed25519.pub").write_text(f"{key}\n")
    if guard:
        tool = hosts / "phastform" / "flatshare" / "src" / "deploy" / "bin" / "node-report"
        tool.parent.mkdir(parents=True)
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
    authorized = hosts / "phastform" / ".ssh" / "authorized_keys"
    authorized.write_text(authorized_keys)
    bin_dir.mkdir()
    (bin_dir / "fake-ssh").write_text(FAKE_SSH)
    (bin_dir / "fake-ssh").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_HOSTS": str(hosts),
        "FLATSHARE_SSH": "fake-ssh",
    }
    done = subprocess.run(
        ["bash", str(DEPLOY / "authorize-guard.sh"), "cyan", "phastform"],
        env=env,
        capture_output=True,
        text=True,
    )
    return done, authorized.read_text(), hosts / "cyan"


GUARD_ENTRY = 'restrict,command="exec flatshare/src/deploy/bin/node-report"'
"""The forced command of the guard key's entry: what a soak reads of the node (its GPU processes and its units'
journal lines), and nothing else."""


def test_the_guard_key_may_only_run_the_nodes_report(tmp_path: Path) -> None:
    # cyan's guard key, made where it has none, gets one entry: no forwarding, no pty, and node-report for every session
    # whatever it asks. An earlier entry of the key goes; every other entry stays as it was.
    tunnel = f'restrict,command="exec nc -q0 -U /run/user/1/flatshare/engine.sock" {KEYS["engine"]}'
    done, authorized, cyan = authorize_guard(tmp_path / "new", f"{LOGIN}\n{tunnel}\n")
    assert done.returncode == 0, done.stderr
    made = (cyan / ".ssh" / "flatshare_guard_ed25519.pub").read_text().split()
    assert made[0] == "ssh-ed25519" and made[2].startswith("flatshare-guard@")
    assert (cyan / ".ssh" / "flatshare_guard_ed25519").exists()
    assert authorized.splitlines() == [LOGIN, tunnel, f"{GUARD_ENTRY} {' '.join(made)}"]
    assert done.stdout.startswith("authorized: the guard key reads what a soak reads of ")
    # A key it already has is kept, and its earlier entry replaced, with or without a comment.
    for case, key in (("again", f"{KEYS['stt']} flatshare-guard@cyan"), ("bare", KEYS["stt"])):
        earlier = f'restrict,command="/bin/true" {KEYS["stt"]} flatshare-guard@cyan'
        done, authorized, cyan = authorize_guard(tmp_path / case, f"{LOGIN}\n{earlier}\n", key=key)
        assert done.returncode == 0, done.stderr
        assert authorized.splitlines() == [LOGIN, f"{GUARD_ENTRY} {key}"], case
        assert (cyan / ".ssh" / "flatshare_guard_ed25519").read_text() == "private\n"


def test_the_guard_key_is_not_authorized_on_a_node_without_its_report(tmp_path: Path) -> None:
    done, authorized, _ = authorize_guard(tmp_path, f"{LOGIN}\n", guard=False, key=KEYS["tts"])
    assert done.returncode == 2 and "has no ~/flatshare/src/deploy/bin/node-report" in done.stderr
    assert authorized == f"{LOGIN}\n"


NODE_JOURNAL = "\n".join(
    json.dumps({"__REALTIME_TIMESTAMP": str(int(at * 1e6)), "USER_UNIT": unit, "MESSAGE": message})
    for at, unit, message in (
        (
            1_800_000_010.5,
            "flatshare-engine.service",
            "flatshare-engine.service: Main process exited, code=killed",
        ),
        (
            1_800_000_011.0,
            "flatshare-engine.service",
            "flatshare-engine.service: Scheduled restart job",
        ),
        (1_800_000_012.0, "other.service", "Started other.service"),
    )
)
"""The node's user manager's lines about its units, as ``journalctl --user -o json`` gives them."""


def node_report(tmp_path: Path, asked: str | None, argv: list[str] = (), status: int = 0):
    """deploy/bin/node-report as the guard key's forced command runs it, its session's command ``asked``, beside a
    stand-in gpu-guard (``status``) and on a node whose systemctl and journalctl are stand-ins that log their
    arguments. Returns the run and the journal's calls."""
    node = tmp_path / "node"
    node.mkdir(parents=True, exist_ok=True)
    (node / "node-report").write_text((DEPLOY / "bin" / "node-report").read_text())
    (node / "gpu-guard").write_text(
        f"#!/bin/sh\nprintf 'ours\\t2211\\tme\\t13990\\tpython3\\n'\nexit {status}\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "journalctl.calls"
    (bin_dir / "systemctl").write_text(
        "#!/bin/sh\nprintf 'flatshare-engine.service loaded active running Flatshare engine\\n"
        "flatshare-tts.service loaded active running Flatshare TTS\\n'\n"
    )
    (tmp_path / "journal.json").write_text(NODE_JOURNAL + "\n")
    (bin_dir / "journalctl").write_text(
        f'#!/bin/sh\necho "$*" >>{calls}\ncase "$*" in\n'
        f'  *"-o json"*) cat {tmp_path / "journal.json"} ;;\n'
        '  *) printf \'CUDA error: an illegal memory access\\n{"message":"request","status":200}\\n'
        "Engine core proc died\\n' ;;\nesac\n"
    )
    for tool in ("systemctl", "journalctl"):
        (bin_dir / tool).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "SSH_ORIGINAL_COMMAND"}
    env["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    if asked is not None:
        env["SSH_ORIGINAL_COMMAND"] = asked
    done = subprocess.run(
        [sys.executable, str(node / "node-report"), *argv], env=env, capture_output=True, text=True
    )
    return done, calls.read_text().splitlines() if calls.exists() else []


def test_the_nodes_report_gives_its_gpus_processes_and_its_units_lines(tmp_path: Path) -> None:
    # What the soak's guard key runs on a GPU node, whatever its session asks: gpu-guard --list, whose status it exits
    # with, and, for a session that asks since=EPOCH, the user manager's lines about the node's flatshare units from
    # then, a failure's with the unit's own last lines (a server's request lines left out).
    done, calls = node_report(tmp_path / "asked", "since=1800000000.5")
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["gpu_guard"] == {
        "status": 0,
        "stdout": "ours\t2211\tme\t13990\tpython3\n",
        "stderr": "",
    }
    assert report["journal"] == [
        {"at": 1_800_000_010.5, "unit": "flatshare-engine.service",
         "message": "flatshare-engine.service: Main process exited, code=killed",
         "own_lines": ["CUDA error: an illegal memory access", "Engine core proc died"]},
        {"at": 1_800_000_011.0, "unit": "flatshare-engine.service",
         "message": "flatshare-engine.service: Scheduled restart job"},
    ]  # fmt: skip
    asked = calls[0].split()
    assert asked[:3] == ["--user", "--since", "@1800000000.500000"]
    assert {"USER_UNIT=flatshare-engine.service", "USER_UNIT=flatshare-tts.service"} <= set(asked)
    assert "_SYSTEMD_USER_UNIT=flatshare-engine.service" in calls[1]
    # A session that asks anything else gets the GPU's processes alone, and the journal is not read.
    for case, asked_for in (("other", "rm -rf ~"), ("none", None), ("shell", "since=1; id")):
        done, calls = node_report(tmp_path / case, asked_for)
        assert done.returncode == 0 and "journal" not in json.loads(done.stdout), case
        assert calls == [], case
    # Run on this host (a single host's guard), the time comes as its argument; the status is gpu-guard's.
    done, _ = node_report(tmp_path / "local", None, ["since=1800000000"], status=1)
    assert done.returncode == 1 and len(json.loads(done.stdout)["journal"]) == 2


def test_cyans_install_makes_the_gateways_secret_and_metrics_key_once(tmp_path: Path) -> None:
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bin_dir.mkdir()
    for tool in ("systemctl", "loginctl"):  # cyan's user manager, which the test leaves alone
        (bin_dir / tool).write_text("#!/bin/sh\n")
        (bin_dir / tool).chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    install = ["bash", str(DEPLOY / "cyan" / "install.sh")]
    subprocess.run(install, env=env, check=True, capture_output=True)
    config = home / ".config" / "flatshare"
    files = [config / "cache-salt-secret", config / "metrics-key"]
    made = [path.read_text() for path in files]
    for path, text in zip(files, made, strict=True):
        assert re.fullmatch(r"[0-9a-f]{64}\n", text), path
        # The gateway refuses a file that another user may open.
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert made[0] != made[1]
    # A new secret would change every tenant's cache salt, a new key every scraper's: the next install keeps them.
    subprocess.run(install, env=env, check=True, capture_output=True)
    assert [path.read_text() for path in files] == made
    assert sorted(p.name for p in config.iterdir()) == ["cache-salt-secret", "metrics-key"]


def test_cyans_install_says_when_the_gateways_binary_is_not_the_sources_version(
    tmp_path: Path,
) -> None:
    # The gateway's binary was built apart from the source it was deployed with, so it lagged a release unnoticed:
    # cyan's DEPLOYED was v1.2 while its binary said 1.1.0. A release checks it before it switches; an install outside
    # one says so and still succeeds.
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bin_dir.mkdir()
    for tool in ("systemctl", "loginctl"):
        (bin_dir / tool).write_text("#!/bin/sh\n")
        (bin_dir / tool).chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    source = tomllib.loads((REPO / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    gateway = home / "flatshare" / "bin" / "flatshare-gateway"
    gateway.parent.mkdir(parents=True)

    def install() -> str:
        done = subprocess.run(
            ["bash", str(DEPLOY / "cyan" / "install.sh")], env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
        return done.stderr

    said = (
        "install: ~/flatshare/bin/flatshare-gateway is {}, the source version {}: "
        "release this commit with its gateway as ops/RUNBOOK.md's Deploy says\n"
    )
    assert install() == said.format("missing", source)
    gateway.write_text("#!/bin/sh\necho 'flatshare-gateway 1.1.0'\n")
    gateway.chmod(0o755)
    assert install() == said.format("version 1.1.0", source)
    gateway.write_text("#!/bin/sh\nexit 126\n")  # a binary for another machine, say
    assert install() == said.format("not runnable", source)
    gateway.write_text(f"#!/bin/sh\necho 'flatshare-gateway {source}'\n")
    assert install() == ""


LINKED_SYSTEMCTL = """\
#!/bin/sh
# systemctl --user ...: logs each call; disable removes a linked unit's file from the user's units, as systemd's does.
echo "$*" >>"$HOME/systemctl.log"
if [ "$2" = disable ]; then
  rm "$HOME/.config/systemd/user/$4"
fi
"""


def test_cyans_install_drops_links_to_units_the_repository_no_longer_has(tmp_path: Path) -> None:
    # Two links left by units the repository dropped (the tunnels before flatshare-tunnel@.service): the
    # first one's disable removed the link itself, its rm then failed, and under set -e the install
    # stopped before the second link, daemon-reload and the enables (the W6 deploy, 2026-10-25).
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(LINKED_SYSTEMCTL)
    (bin_dir / "loginctl").write_text("#!/bin/sh\n")
    for tool in ("systemctl", "loginctl"):
        (bin_dir / tool).chmod(0o755)
    gone = ["flatshare-tunnel-gray.service", "flatshare-tunnel-phastform.service"]
    for name in gone:
        (units / name).symlink_to(DEPLOY / "cyan" / "systemd" / name)  # no such unit now
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    done = subprocess.run(
        ["bash", str(DEPLOY / "cyan" / "install.sh")], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert not any(os.path.lexists(units / name) for name in gone)
    calls = (home / "systemctl.log").read_text().splitlines()
    assert [c for c in calls if " disable " in c] == [f"--user disable -q {name}" for name in gone]
    assert calls[-2:] == [
        "--user daemon-reload",
        "--user enable -q flatshare-voicenet.service flatshare-livekit.service flatshare-tunnel@phastform.service "
        "flatshare-voice-worker.service",
    ]
    linked = {p.name for p in units.iterdir()}
    assert linked == {p.name for p in (DEPLOY / "cyan" / "systemd").iterdir()}  # the units it has


FAKE_SYSTEMCTL = """\
#!/bin/sh
# systemctl --user ...: a GPU node's user manager, every unit running; logs each call that changes something.
shift
if [ "$1" = is-active ]; then
  shift
  [ "$1" = -q ] || for unit; do echo active; done
  exit 0
fi
echo "$*" >>"$HOME/systemctl.log"
"""


def test_the_engine_serves_its_development_routes_only_with_the_measurement_drop_in(
    tmp_path: Path,
) -> None:
    unit = (DEPLOY / "phastform" / "systemd" / "flatshare-engine.service").read_text()
    assert "VLLM_SERVER_DEV_MODE" not in unit
    measure = DEPLOY / "phastform" / "measure" / "flatshare-engine.conf"
    assert "Environment=VLLM_SERVER_DEV_MODE=1" in measure.read_text().splitlines()

    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(FAKE_SYSTEMCTL)
    (bin_dir / "systemctl").chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    drop_in = home / ".config" / "systemd" / "user" / "flatshare-engine.service.d" / "measure.conf"
    log = home / "systemctl.log"

    def variant(*args: str) -> str:
        done = subprocess.run(
            ["bash", str(DEPLOY / "phastform" / "variant.sh"), *args],
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout

    # On: linked, and the running engine restarts to take it; asking again changes nothing.
    variant("measure", "on")
    assert drop_in.resolve() == measure
    assert log.read_text().splitlines() == [
        "daemon-reload",
        "reset-failed flatshare-engine",
        "restart flatshare-engine",
    ]
    log.unlink()
    assert variant("measure", "on") == "variant: the measurement drop-in is already on\n"
    assert not log.exists()
    # Switching the engine's variant keeps it.
    variant("use", "pr54530")
    assert drop_in.resolve() == measure
    assert (drop_in.parent / "variant-pr54530.conf").exists()
    variant("use", "frozen")
    assert drop_in.resolve() == measure
    # Off: removed, and the engine restarts without the development routes.
    log.unlink()
    variant("measure", "off")
    assert not drop_in.exists()
    assert log.read_text().splitlines() == [
        "daemon-reload",
        "reset-failed flatshare-engine",
        "restart flatshare-engine",
    ]


STATEFUL_SYSTEMCTL = """\
#!/bin/sh
# systemctl --user ...: a GPU node's user manager whose units' states are files in $HOME/state, every call logged. A
# start skips a unit whose drop-ins hold ConditionPathExists=!/ (its condition never holds), as systemd does, and the
# target starts the three GPU servers; is-active with several units succeeds when one is active, as systemctl does.
shift
command=$1
shift
state=$HOME/state
units=$HOME/.config/systemd/user
mkdir -p "$state"
echo "$command $*" >>"$HOME/systemctl.log"
skipped() { cat "$units/$1.service.d"/*.conf 2>/dev/null | grep -qx 'ConditionPathExists=!/'; }
case $command in
  is-active)
    quiet=""
    if [ "$1" = -q ]; then quiet=1; shift; fi
    status=3
    for unit; do
      if [ -e "$state/${unit%.service}" ]; then status=0; word=active; else word=inactive; fi
      [ -n "$quiet" ] || echo "$word"
    done
    exit $status
    ;;
  start | restart)
    for unit; do
      unit=${unit%.service}
      if [ "$unit" = flatshare.target ]; then
        for server in flatshare-engine flatshare-stt flatshare-tts; do skipped $server || touch "$state/$server"; done
      else
        skipped "$unit" || touch "$state/$unit"
      fi
    done
    ;;
  stop)
    for unit; do rm -f "$state/${unit%.service}"; done
    ;;
esac
exit 0
"""


def test_variant_sh_runs_the_agents_only_card_without_its_speech_servers_and_restores_them(
    tmp_path: Path,
) -> None:
    # A dedicated background card runs the engine alone: agents-only keeps the STT and the TTS from starting, the box
    # counts as running with them stopped (and only so), and the variant after it starts them again.
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    (home / "state").mkdir(parents=True)
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(STATEFUL_SYSTEMCTL)
    (bin_dir / "systemctl").chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    servers = ("flatshare-engine", "flatshare-stt", "flatshare-tts")
    for server in servers:
        (home / "state" / server).touch()  # the frozen box runs
    units = home / ".config" / "systemd" / "user"

    def variant(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(DEPLOY / "phastform" / "variant.sh"), *args],
            env=env,
            capture_output=True,
            text=True,
        )

    def running() -> set[str]:
        return {path.name for path in (home / "state").iterdir()}

    done = variant("use", "agents-only")
    assert done.returncode == 0, done.stderr
    assert running() == {"flatshare-engine"}
    for server in servers:
        link = units / f"{server}.service.d" / "variant-agents-only.conf"
        assert (
            link.resolve() == DEPLOY / "phastform" / "variants" / "agents-only" / f"{server}.conf"
        )
    assert (
        "stop flatshare.target flatshare-engine flatshare-stt flatshare-tts"
        in (home / "systemctl.log").read_text().splitlines()
    )
    assert variant("use", "agents-only").stdout == "variant: the box already runs agents-only \n"
    # A speech server started by hand beside it is no box that runs agents-only: the switch stops it.
    (home / "state" / "flatshare-stt").touch()
    done = variant("use", "agents-only")
    assert done.returncode == 0 and running() == {"flatshare-engine"}, done.stderr
    # Flags over it restart the engine alone; the speech servers stay off.
    done = variant("use", "agents-only", "--max-num-batched-tokens", "1024")
    assert done.returncode == 0 and running() == {"flatshare-engine"}, done.stderr
    # The frozen box again: the drop-ins go and the three servers run.
    done = variant("use", "frozen")
    assert done.returncode == 0, done.stderr
    assert running() == set(servers)
    assert not list(units.glob("*.service.d/variant-*.conf"))
    # on|off add and remove it the same way.
    assert variant("on", "agents-only").returncode == 0 and running() == {"flatshare-engine"}
    assert variant("off", "agents-only").returncode == 0 and running() == set(servers)


def test_the_agents_only_cards_pool_takes_the_speech_servers_memory_and_fits_the_card() -> None:
    # Its pool is the box's 5 GiB and the 3.4 GiB the STT and the TTS hold (ARCHITECTURE.md 3.1: about 2.4 and 1.0),
    # 3,822 blocks of 16 tokens at Qwen3-4B's 144 KiB a token; the engine, about 10.6 GiB with the box's pool, then
    # takes 14.0 GiB of the card, a budget of 0.88 under R2's ceiling, which vLLM wants free at the start and which a
    # card with the speech servers resident does not have.
    conf = (DEPLOY / "phastform" / "variants" / "agents-only" / "flatshare-engine.conf").read_text()
    [start] = [line for line in conf.splitlines() if line.startswith("ExecStart=%h")]
    assert start.startswith("ExecStart=%h/flatshare/src/deploy/bin/in-private-net ")
    flags = dict(re.findall(r"(--[a-z-]+) ([^ $-][^ ]*)", start.split("engine.yaml", 1)[1]))
    gib = 1 << 30
    box = int(
        re.search(
            r"^kv-cache-memory: (\d+)$", (DEPLOY / "phastform" / "engine.yaml").read_text(), re.M
        )[1]
    )
    assert box == 5 * gib
    architecture = (REPO / "docs" / "ARCHITECTURE.md").read_text()
    stt, tts = (
        float(re.search(rf"^\| {name} \|.*\| ≈ ([\d.]+) GiB \|$", architecture, re.M)[1])
        for name in ("STT", "TTS")
    )
    engine = float(re.search(r"^\| LLM \|.*\| ≈ ([\d.]+) GiB \|$", architecture, re.M)[1])
    assert (stt, tts, engine) == (2.4, 1.0, 10.6)
    pool = int(flags["--kv-cache-memory"])
    assert pool == int((5 + stt + tts) * gib)
    token = 36 * 8 * 128 * 2 * 2  # layers x KV heads x head size x (K, V) x BF16 bytes: 144 KiB
    assert (pool // (token * 16), pool // (token * 16) * 16) == (3822, 61152)
    assert box // (token * 16) * 16 == 36400  # the box's, as serving.json records it
    loader = importlib.machinery.SourceFileLoader("profile_rules", str(DEPLOY / "profile.py"))
    rules = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("profile_rules", loader)
    )
    loader.exec_module(rules)
    budget = engine + stt + tts
    assert rules.utilization(budget, 16311) == flags["--gpu-memory-utilization"] == "0.88"
    assert float(flags["--gpu-memory-utilization"]) <= rules.CARD_SHARE
    # Free at the start, vLLM wants 0.88 of the 15,852 MiB it counts; the speech servers' 3.4 GiB beside an otherwise
    # empty card leave less than that.
    assert 15852 - (stt + tts) * 1024 < 0.88 * 15852 < 15852 - 600
    assert (flags["--scheduling-policy"], flags["--max-num-batched-tokens"]) == ("fcfs", "2048")
    assert flags["--long-prefill-token-threshold"] == "0"


def test_the_agents_only_cards_step_cap_follows_what_e2_measured() -> None:
    # The variant's comment cites E2's discriminating scenario: the agents' tokens/s at a step cap of 1,024 with no
    # threshold against the frozen box's 512/256, the medians of their five runs in W4's records.
    import statistics

    results = REPO / "benchmarks" / "results" / "2026-10-23-e2"
    goodput = {}
    for point in ("disc-a4-b1024-t0-s24", "disc-frozen-s24"):
        runs = [json.loads(path.read_text()) for path in results.glob(f"*/{point}/summary.json")]
        assert len(runs) == 5, point
        goodput[point] = statistics.median(s["background"]["tokens_per_s"] for s in runs)
    larger, box = goodput["disc-a4-b1024-t0-s24"], goodput["disc-frozen-s24"]
    assert (round(larger), round(box), round(100 * (larger / box - 1))) == (4097, 3710, 10)
    conf = (DEPLOY / "phastform" / "variants" / "agents-only" / "flatshare-engine.conf").read_text()
    assert "10% more (4,097 against 3,710 tokens a second" in conf.replace("\n# ", " ")


class _Health(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


def test_wait_healthy_reaches_a_server_on_its_unix_socket() -> None:
    wait = ["bash", str(DEPLOY / "bin" / "wait-healthy")]
    socket = socket_file("flatshare-wait-", "engine.sock")
    server = serve(UnixServer(str(socket), _Health))
    try:
        done = subprocess.run(
            [*wait, "--unix-socket", str(socket), "http://localhost/health", "5"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith(f"wait-healthy: http://localhost/health on {socket} healthy")
    finally:
        stop(server)
    # Nobody serves the socket now: it gives up after its timeout.
    done = subprocess.run(
        [*wait, "--unix-socket", str(socket), "http://localhost/health", "0"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1
    assert f"on {socket} not healthy after 0s" in done.stderr
    # --pid=PID, the unit's main process ($MAINPID): an empty one is no PID, as before.
    done = subprocess.run(
        [*wait, "--unix-socket", str(socket), "--pid=", "http://localhost/health", "0"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1 and "not healthy after 0s" in done.stderr


def test_wait_healthy_gives_up_once_the_server_it_waits_for_is_gone() -> None:
    # A server that dies in its start (a model that does not fit, a library that does not load) fails the start at
    # once, not after the unit's whole timeout (1,200 s for the engine), whether it was reaped already or is a
    # zombie its supervisor has not reaped yet (flatshare-svc's runner reaps its main process only after ExecStartPost).
    wait = ["bash", str(DEPLOY / "bin" / "wait-healthy")]
    nobody = "http://127.0.0.1:9/health"  # the discard port: never healthy
    gone = subprocess.Popen(["true"])
    gone.wait()
    zombie = subprocess.Popen(["true"])
    time.sleep(0.5)  # it has exited, unreaped
    try:
        for pid in (gone.pid, zombie.pid):
            start = time.monotonic()
            done = subprocess.run(
                [*wait, f"--pid={pid}", nobody, "60"], capture_output=True, text=True
            )
            assert done.returncode == 1, done.stderr
            assert f"the server (pid {pid}) exited before {nobody} was healthy" in done.stderr
            assert time.monotonic() - start < 10
    finally:
        zombie.wait()
    alive = subprocess.Popen(["sleep", "30"])
    try:
        done = subprocess.run(
            [*wait, f"--pid={alive.pid}", nobody, "0"], capture_output=True, text=True
        )
        assert (
            done.returncode == 1 and "not healthy after 0s" in done.stderr
        )  # its own timeout, as before
    finally:
        alive.kill()
        alive.wait()


FAKE_SS = """\
#!/bin/sh
# ss -H -lntup | -lxp: the listeners in $FAKE_SS, as ss prints them.
case $2 in
  -lntup) cat "$FAKE_SS/inet" ;;
  -lxp) cat "$FAKE_SS/unix" ;;
esac
"""


HOST_NS = "net:[4026531840]"
"""The host's network namespace, as /proc/PID/ns/net names it."""
UNIX_TABLE = "Num       RefCount Protocol Flags    Type St Inode Path\n"
TCP_TABLE = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"


class Listeners:
    """A host for audit-listeners: what ``ss`` lists, and the processes' cgroups and commands under a
    fake /proc. Each process of ours is in a flatshare-*.service unit. The audit's own network namespace is the
    host's; a process in a namespace of its own (``namespaced``) holds sockets that ss, in the host's, does
    not list, and that its namespace's tables under /proc/PID/net do."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.runtime = root / "run"
        (self.runtime / "flatshare").mkdir(parents=True)
        (self.runtime / "flatshare").chmod(0o700)
        self.ss = root / "ss"
        self.ss.mkdir()
        self.proc = root / "proc"
        self.inet: list[str] = []
        self.unix: list[str] = []
        self.pids = 0
        self.inodes = 500000
        self.spaces: dict[str, dict[str, list[str]]] = {}  # namespace: its tables' rows
        self.members: dict[str, list[str]] = {}  # namespace: its processes
        (self.proc / "self" / "ns").mkdir(parents=True)
        (self.proc / "self" / "ns" / "net").symlink_to(HOST_NS)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "ss").write_text(FAKE_SS)
        (bin_dir / "ss").chmod(0o755)
        self.env = os.environ | {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_SS": str(self.ss),
            "FLATSHARE_AUDIT_PROC": str(self.proc),
            "XDG_RUNTIME_DIR": str(self.runtime),
        }

    def process(
        self,
        unit: str | None,
        command: str = "vllm serve",
        flatshare_svc: bool = False,
        env: dict[str, str] | None = None,
        parent: str | None = None,
        overwritten: bool = False,
    ) -> str:
        """A process, in ``unit`` or (None) another user's session; returns ss's note of it. Under systemd a
        unit's process is in the unit's cgroup; under deploy/bin/flatshare-svc (``flatshare_svc``) it stays in the SSH
        session's, and flatshare-svc names its unit in its environment (FLATSHARE_SVC_UNIT), beside ``env``. ``parent``
        is its parent's pid; ``overwritten``, an environment that /proc shows as NULs alone, as vLLM's engine core's
        once setproctitle has blanked it."""
        self.pids += 1
        pid = str(1000 + self.pids)
        (self.proc / pid).mkdir(parents=True)
        if parent:
            (self.proc / pid / "status").write_text(f"Name:\tVLLM::EngineCor\nPPid:\t{parent}\n")
        if flatshare_svc:
            cgroup = "/user.slice/user-1007.slice/session-4.scope"
        elif unit:
            cgroup = f"/user.slice/user-1007.slice/user@1007.service/app.slice/{unit}"
        else:
            cgroup = "/init.scope"
        (self.proc / pid / "cgroup").write_text(f"0::{cgroup}\n")
        (self.proc / pid / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        environ = (
            {"HOME": "/home/flatshare"}
            | ({"FLATSHARE_SVC_UNIT": unit} if flatshare_svc and unit else {})
            | (env or {})
        )
        (self.proc / pid / "environ").write_bytes(
            b"\0" * 64
            if overwritten
            else b"".join(f"{k}={v}\0".encode() for k, v in environ.items())
        )
        return f'users:(("python3",pid={pid},fd=3))'

    def namespaced(
        self, unit: str, space: str, sockets: list[str], env: dict[str, str] | None = None
    ) -> str:
        """A process of ``unit`` in network namespace ``space`` holding ``sockets``: each a path, an @NAME
        (an abstract socket) or a TCP port on 127.0.1.1, as an engine core's gloo listeners bind. Returns its
        pid."""
        [pid] = re.findall(r"pid=(\d+)", self.process(unit, env=env))
        (self.proc / pid / "ns").mkdir()
        (self.proc / pid / "ns" / "net").symlink_to(space)
        (self.proc / pid / "fd").mkdir()
        tables = self.spaces.setdefault(space, {"unix": [], "tcp": []})
        self.members.setdefault(space, []).append(pid)
        for fd, address in enumerate(sockets, start=3):
            self.inodes += 1
            (self.proc / pid / "fd" / str(fd)).symlink_to(f"socket:[{self.inodes}]")
            if address.isdigit():
                tables["tcp"].append(
                    f"   0: 0101007F:{int(address):04X} 00000000:0000 0A 00000000:00000000 00:00000000 "
                    f"00000000  1018        0 {self.inodes} 1 0000000000000000 100 0 0 10 0"
                )
            else:
                tables["unix"].append(
                    f"0000000000000000: 00000002 00000000 00010000 0001 01 {self.inodes} {address}"
                )
        return pid

    def manager(self) -> str:
        """The user manager, systemd --user, which holds socket units' sockets; returns ss's note of it."""
        self.pids += 1
        pid = str(1000 + self.pids)
        (self.proc / pid).mkdir(parents=True)
        (self.proc / pid / "cgroup").write_text(
            "0::/user.slice/user-1007.slice/user@1007.service/init.scope\n"
        )
        (self.proc / pid / "cmdline").write_bytes(b"/lib/systemd/systemd\0--user")
        # The host's namespace, which the audit takes as such.
        (self.proc / pid / "ns").mkdir()
        (self.proc / pid / "ns" / "net").symlink_to(HOST_NS)
        return f'users:(("systemd",pid={pid},fd=40))'

    def tcp(self, address: str, unit: str | None, command: str = "vllm serve", **how) -> None:
        held = (
            self.process(unit, command, **how) if unit else ""
        )  # ss shows no process of another user's
        self.inet.append(f"tcp LISTEN 0 4096 {address} 0.0.0.0:* {held}")

    def udp(self, address: str, unit: str | None, command: str = "livekit-server", **how) -> None:
        held = self.process(unit, command, **how) if unit else ""
        self.inet.append(f"udp UNCONN 0 0 {address} 0.0.0.0:* {held}")

    def socket(self, path: str, unit: str | None, **how) -> None:
        held = self.process(unit, **how) if unit else ""
        self.unix.append(f"u_str LISTEN 0 4096 {path} 362417538 * 0 {held}")

    def audit(self, *args: str) -> subprocess.CompletedProcess[str]:
        (self.ss / "inet").write_text("".join(f"{line}\n" for line in self.inet))
        (self.ss / "unix").write_text("".join(f"{line}\n" for line in self.unix))
        for space, tables in self.spaces.items():
            for pid in self.members[space]:
                net = self.proc / pid / "net"
                net.mkdir(exist_ok=True)
                (net / "unix").write_text(
                    UNIX_TABLE + "".join(f"{row}\n" for row in tables["unix"])
                )
                (net / "tcp").write_text(TCP_TABLE + "".join(f"{row}\n" for row in tables["tcp"]))
        return subprocess.run(
            [sys.executable, str(DEPLOY / "bin" / "audit-listeners"), *args],
            env=self.env,
            capture_output=True,
            text=True,
        )


def test_the_audit_passes_a_gpu_node_whose_servers_listen_on_their_sockets(tmp_path: Path) -> None:
    host = Listeners(tmp_path)
    socket_dir = host.runtime / "flatshare"
    for unit, name in (
        ("flatshare-engine", "engine"),
        ("flatshare-stt", "stt"),
        ("flatshare-tts", "tts"),
    ):
        host.socket(f"{socket_dir}/{name}.sock", f"{unit}.service")
    host.socket(
        f"{socket_dir}/8c3f0b2e-0d2c-4a51-9d55-0f6a2c0b7f3e", "flatshare-engine.service"
    )  # vLLM's ZMQ IPC
    host.tcp("127.0.0.1:631", None)  # another user's
    host.socket("/run/systemd/private", None)
    sockets = [f"{socket_dir}/{name}.sock" for name in ("engine", "stt", "tts")]
    done = host.audit(*(arg for path in sockets for arg in ("--expect", path)))
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        f"audit-listeners: ok: the box's units hold 4 Unix sockets, all in {host.runtime}, "
        "and 0 TCP or UDP listeners, all allowed\n"
    )


def test_the_audit_fails_on_any_listener_of_ours_outside_the_allowed_set(tmp_path: Path) -> None:
    host = Listeners(tmp_path)
    # Today's box: each server on a loopback port that every user of the host reaches.
    host.tcp("127.0.0.1:8000", "flatshare-engine.service")
    host.tcp("127.0.0.1:8100", "flatshare-stt.service")
    host.tcp("127.0.0.1:8880", "flatshare-tts.service")
    host.socket("/tmp/8c3f0b2e-0d2c-4a51-9d55-0f6a2c0b7f3e", "flatshare-engine.service")
    host.socket("@flatshare-abstract", "flatshare-tts.service")
    host.runtime.joinpath("flatshare").chmod(0o755)
    done = host.audit("--expect", f"{host.runtime}/flatshare/engine.sock")
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        f"audit-listeners: {host.runtime}/flatshare is mode 755, owned by uid {os.getuid()}: it must be 0700 and ours",
        "audit-listeners: flatshare-engine.service listens on tcp 127.0.0.1:8000",
        "audit-listeners: flatshare-stt.service listens on tcp 127.0.0.1:8100",
        "audit-listeners: flatshare-tts.service listens on tcp 127.0.0.1:8880",
        "audit-listeners: flatshare-engine.service listens on a socket outside "
        f"{host.runtime}: /tmp/8c3f0b2e-0d2c-4a51-9d55-0f6a2c0b7f3e",
        "audit-listeners: flatshare-tts.service listens on an abstract socket outside "
        f"{host.runtime}: @flatshare-abstract",
        f"audit-listeners: nothing listens on {host.runtime}/flatshare/engine.sock",
    ]


def test_the_audit_allows_the_gateways_port_and_an_sglang_engine_on_loopback(
    tmp_path: Path,
) -> None:
    host = Listeners(tmp_path)
    host.tcp("127.0.0.1:18080", "flatshare-gateway.service")
    host.tcp("127.0.0.1:8000", "flatshare-engine.service", "python -m sglang.launch_server")
    host.socket(
        f"{host.runtime}/flatshare/phastform-engine.sock", "flatshare-tunnel@phastform.service"
    )
    assert host.audit().returncode == 0
    # Beyond loopback, and under any other unit, the same addresses fail.
    host.tcp("0.0.0.0:29500", "flatshare-engine.service", "python -m sglang.launch_server")
    host.tcp("127.0.0.1:18000", "flatshare-tunnel-phastform.service", "ssh -N")
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        "audit-listeners: flatshare-engine.service listens on tcp 0.0.0.0:29500",
        "audit-listeners: flatshare-tunnel-phastform.service listens on tcp 127.0.0.1:18000",
    ]


def test_the_audit_takes_the_user_managers_hold_on_the_gateways_port_for_flatshare_gateway_socket(
    tmp_path: Path,
) -> None:
    host = Listeners(tmp_path)
    # flatshare-gateway.socket's port, held by the user manager and the gateway that inherited it; then by the manager
    # alone, while no gateway runs. Both are the gateway's.
    gateway = host.process("flatshare-gateway.service", "flatshare-gateway --config gateway.toml")
    both = f"{host.manager()[:-1]},{gateway.removeprefix('users:(')}"
    host.inet.append(f"tcp LISTEN 0 1024 127.0.0.1:18080 0.0.0.0:* {both}")
    host.socket(f"{host.runtime}/flatshare/gateway.sock", "flatshare-gateway.service")
    assert host.audit().returncode == 0
    host.inet = [f"tcp LISTEN 0 1024 127.0.0.1:18080 0.0.0.0:* {host.manager()}"]
    done = host.audit()
    assert done.returncode == 0, done.stderr
    # Another socket unit of the user's, elsewhere, is not ours to judge.
    host.inet.append(f"tcp LISTEN 0 128 127.0.0.1:9100 0.0.0.0:* {host.manager()}")
    assert host.audit().returncode == 0


def test_the_audit_fails_when_another_user_holds_the_gateways_port(tmp_path: Path) -> None:
    host = Listeners(tmp_path)
    host.tcp("127.0.0.1:18080", None)
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr == (
        "audit-listeners: 127.0.0.1:18080 is held by a process that is not flatshare-gateway.service's: "
        "another user's?\n"
    )


def test_the_audit_tells_flatshare_svcs_units_by_the_unit_their_environment_names(
    tmp_path: Path,
) -> None:
    # Under deploy/bin/flatshare-svc a unit's processes stay in the SSH session's cgroup, and the gateway binds its own
    # listeners (no socket unit holds them): flatshare-svc's FLATSHARE_SVC_UNIT in a process's environment names its
    # unit, as gpu-guard reads it, so the gateway's port is the gateway's, and the GPU servers' sockets are judged.
    host = Listeners(tmp_path)
    socket_dir = host.runtime / "flatshare"
    host.tcp(
        "127.0.0.1:18080",
        "flatshare-gateway.service",
        "flatshare-gateway --config gateway.toml",
        flatshare_svc=True,
    )
    host.socket(f"{socket_dir}/gateway.sock", "flatshare-gateway.service", flatshare_svc=True)
    for unit, name in (
        ("flatshare-engine", "engine"),
        ("flatshare-stt", "stt"),
        ("flatshare-tts", "tts"),
    ):
        host.socket(f"{socket_dir}/{name}.sock", f"{unit}.service", flatshare_svc=True)
    host.tcp("127.0.0.1:631", None)  # another user's
    done = host.audit("--expect", f"{socket_dir}/engine.sock")
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        f"audit-listeners: ok: the box's units hold 4 Unix sockets, all in {host.runtime}, "
        "and 1 TCP or UDP listeners, all allowed\n"
    )
    # And a unit of flatshare-svc's fails where it would under systemd: on a port, or a socket outside the directory.
    host.tcp("127.0.0.1:8000", "flatshare-engine.service", flatshare_svc=True)
    host.socket("/tmp/flatshare-stt.sock", "flatshare-stt.service", flatshare_svc=True)
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        "audit-listeners: flatshare-engine.service listens on tcp 127.0.0.1:8000",
        f"audit-listeners: flatshare-stt.service listens on a socket outside {host.runtime}: /tmp/flatshare-stt.sock",
    ]


def test_the_audit_tells_an_engine_cores_unit_and_mark_by_its_server_once_its_title_overwrote_them(
    tmp_path: Path,
) -> None:
    # vLLM's engine core blanks its /proc/PID/environ when it sets its title (setproctitle), where flatshare-svc's
    # FLATSHARE_SVC_UNIT and the host's FLATSHARE_PRIVATE_NET=host stood; its parent, the server it serves, keeps them.
    # Under flatshare-svc the audit took the engine core for no unit's and passed its listeners unjudged; on a host
    # whose GPU servers run in the host's network namespace it found the core unmarked and refused gloo's listener (the
    # rehearsal on the box found the same of gpu-guard). Its parent's unit and mark are now its own.
    host = Listeners(tmp_path)
    [server] = re.findall(
        r"pid=(\d+)", host.process("flatshare-engine.service", flatshare_svc=True)
    )
    host.tcp(
        "0.0.0.0:29500", "flatshare-engine.service", "VLLM::EngineCore", flatshare_svc=True, parent=server,
        overwritten=True,
    )  # fmt: skip
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        "audit-listeners: flatshare-engine.service listens on tcp 0.0.0.0:29500"
    ]
    host = Listeners(tmp_path / "marked")
    marked = {"FLATSHARE_PRIVATE_NET": "host"}
    [server] = re.findall(r"pid=(\d+)", host.process("flatshare-engine.service", env=marked))
    host.tcp(
        "127.0.1.1:36087",
        "flatshare-engine.service",
        "VLLM::EngineCore",
        parent=server,
        overwritten=True,
    )
    host.socket(
        "@cuda-uvmfd-4026531836-4049952@",
        "flatshare-engine.service",
        parent=server,
        overwritten=True,
    )
    done = host.audit()
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        f"audit-listeners: ok: the box's units hold 0 Unix sockets, all in {host.runtime}, "
        "and 0 TCP or UDP listeners, all allowed; the GPU servers run in the host's network namespace "
        "(FLATSHARE_PRIVATE_NET=host) with 1 loopback TCP listeners and 1 abstract sockets\n"
    )


def test_the_audit_takes_a_mark_from_no_parent_outside_the_unit(tmp_path: Path) -> None:
    # Under flatshare-svc a unit's main process is the runner's child, whose environment is the shell's that ran
    # systemctl: a mark exported there (FLATSHARE_PRIVATE_NET=host, say, while trying in-private-net by hand) is not the
    # unit's. An unmarked engine's core, its environment blanked by its title, still has its loopback listener judged.
    host = Listeners(tmp_path)
    [runner] = re.findall(
        r"pid=(\d+)",
        host.process(
            None, "flatshare-svc run", flatshare_svc=True, env={"FLATSHARE_PRIVATE_NET": "host"}
        ),
    )
    [server] = re.findall(
        r"pid=(\d+)", host.process("flatshare-engine.service", flatshare_svc=True, parent=runner)
    )
    host.tcp(
        "127.0.1.1:36087", "flatshare-engine.service", "VLLM::EngineCore", flatshare_svc=True, parent=server,
        overwritten=True,
    )  # fmt: skip
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        "audit-listeners: flatshare-engine.service listens on tcp 127.0.1.1:36087"
    ]


def test_the_audit_accepts_the_voice_plane_in_the_hosts_namespace_where_in_voicenet_ran_it_there(
    tmp_path: Path,
) -> None:
    # On a host whose kernel refuses unprivileged user namespaces the voice plane runs in the host's network
    # namespace (FLATSHARE_VOICENET=host, the preflight's degraded path), and in-voicenet marks its processes so: there
    # its listeners show, where in its own namespace they do not. Its UDP sockets (the WebRTC candidates, on the
    # host's interfaces), its loopback TCP listeners and its file sockets are accepted, under either supervisor.
    host = Listeners(tmp_path)
    marked = {"FLATSHARE_VOICENET": "host"}
    host.tcp(
        "127.0.0.1:17880",
        "flatshare-livekit.service",
        "livekit-server",
        flatshare_svc=True,
        env=marked,
    )
    host.udp(
        "127.0.0.1:17882",
        "flatshare-livekit.service",
        "livekit-server",
        flatshare_svc=True,
        env=marked,
    )
    host.udp(
        "10.0.0.5:51234", "flatshare-voice-worker.service", "flatshare-voice-worker", env=marked
    )
    host.tcp(
        "127.0.0.1:41003", "flatshare-voice-worker.service", "flatshare-voice-worker", env=marked
    )
    host.socket(
        "/tmp/pymp-k2j/listener-q8w", "flatshare-voice-worker.service", env=marked
    )  # its forkserver's
    host.tcp(
        "127.0.0.1:18080", "flatshare-gateway.service", "flatshare-gateway", flatshare_svc=True
    )
    done = host.audit()
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        f"audit-listeners: ok: the box's units hold 0 Unix sockets, all in {host.runtime}, "
        "and 1 TCP or UDP listeners, all allowed; the voice plane runs in the host's network namespace "
        "(FLATSHARE_VOICENET=host) with 5 listeners: 2 UDP sockets, 2 loopback TCP listeners and 1 file socket\n"
    )
    # Unmarked, the same listeners are refused; marked, a voice-plane listener beyond what the degraded path
    # allows (TCP off loopback, an abstract socket, which any user of the host's namespace reaches) is refused,
    # and a unit of the GPU node's is no voice plane, marked or not.
    host = Listeners(tmp_path / "unmarked")
    host.tcp("127.0.0.1:17880", "flatshare-livekit.service", "livekit-server", flatshare_svc=True)
    host.udp("10.0.0.5:51234", "flatshare-voice-worker.service", "flatshare-voice-worker")
    host.tcp("0.0.0.0:7881", "flatshare-livekit.service", "livekit-server", env=marked)
    host.socket("@livekit-rtc", "flatshare-voice-worker.service", env=marked)
    host.tcp("127.0.0.1:8000", "flatshare-engine.service", env=marked)
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        "audit-listeners: flatshare-livekit.service listens on tcp 127.0.0.1:17880",
        "audit-listeners: flatshare-voice-worker.service listens on udp 10.0.0.5:51234",
        "audit-listeners: flatshare-livekit.service listens on tcp 0.0.0.0:7881",
        "audit-listeners: flatshare-engine.service listens on tcp 127.0.0.1:8000",
        f"audit-listeners: flatshare-voice-worker.service listens on an abstract socket outside {host.runtime}: "
        "@livekit-rtc",
    ]


def test_in_voicenet_marks_what_it_runs_in_the_hosts_namespace(tmp_path: Path) -> None:
    socket_dir = tmp_path / "flatshare"
    socket_dir.mkdir()
    (socket_dir / "voicenet.pid").write_text("host\n")
    show = [sys.executable, "-c", "import os; print(os.environ.get('FLATSHARE_VOICENET'))"]
    done = subprocess.run(
        ["bash", str(DEPLOY / "bin" / "in-voicenet"), *show],
        env={k: v for k, v in os.environ.items() if k != "FLATSHARE_VOICENET"}
        | {"XDG_RUNTIME_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0 and done.stdout.strip() == "host", done.stderr


def test_the_audit_reads_the_gpu_servers_listeners_in_their_own_network_namespaces(
    tmp_path: Path,
) -> None:
    # in-private-net runs each GPU server in a network namespace of its own, which ss in the host's does not list:
    # the engine cores' gloo listeners and CUDA's abstract sockets there are no other user's to reach, and are
    # accepted; the servers' sockets are files, which every namespace reaches, so they are judged as in the host's,
    # and found for --expect.
    host = Listeners(tmp_path)
    socket_dir = host.runtime / "flatshare"
    engine = "net:[4026532864]"
    host.namespaced(
        "flatshare-engine.service", engine, [f"{socket_dir}/engine.sock", "36087", "34265"]
    )
    host.namespaced(
        "flatshare-engine.service", engine, ["@cuda-uvmfd-4026531836-4049952@", "41133"]
    )
    host.namespaced(
        "flatshare-stt.service",
        "net:[4026532901]",
        [f"{socket_dir}/stt.sock", "38043", "@cuda-uvmfd-1"],
    )
    host.namespaced(
        "flatshare-tts.service", "net:[4026533065]", [f"{socket_dir}/tts.sock", "@cuda-uvmfd-2"]
    )
    host.socket(
        f"{socket_dir}/gateway.sock", "flatshare-gateway.service"
    )  # in the host's namespace, as before
    sockets = [f"{socket_dir}/{name}.sock" for name in ("engine", "stt", "tts")]
    done = host.audit(*(arg for path in sockets for arg in ("--expect", path)))
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        f"audit-listeners: ok: the box's units hold 4 Unix sockets, all in {host.runtime}, "
        "and 0 TCP or UDP listeners, all allowed; flatshare-engine, flatshare-stt, flatshare-tts in network "
        "namespaces of their "
        "own, with 4 TCP or UDP listeners and 3 abstract sockets there\n"
    )
    # A file socket outside the directory is any user's to reach from any namespace: refused, as in the host's.
    # A socket of the namespace's table that no process of ours holds is not judged.
    host.namespaced("flatshare-stt.service", "net:[4026532901]", ["/tmp/flatshare-stt-debug.sock"])
    host.spaces["net:[4026532901]"]["unix"].append(
        f"0000000000000000: 00000002 00000000 00010000 0001 01 999999 {tmp_path}/not-ours.sock"
    )
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr == (
        f"audit-listeners: flatshare-stt.service listens on a socket outside {host.runtime}: "
        "/tmp/flatshare-stt-debug.sock\n"
    )


def test_the_audit_accepts_the_gpu_servers_in_the_hosts_namespace_where_the_host_marks_them(
    tmp_path: Path,
) -> None:
    # Where the kernel refuses unprivileged user namespaces the host's drop-ins mark the GPU servers
    # FLATSHARE_PRIVATE_NET=host and in-private-net runs them in the host's network namespace: there their engine cores'
    # loopback TCP listeners on ephemeral ports (gloo's) and CUDA's abstract sockets are accepted, and counted apart.
    host = Listeners(tmp_path)
    socket_dir = host.runtime / "flatshare"
    marked = {"FLATSHARE_PRIVATE_NET": "host"}
    host.tcp("127.0.1.1:36087", "flatshare-engine.service", env=marked)
    host.tcp("127.0.1.1:38043", "flatshare-stt.service", env=marked)
    host.socket("@cuda-uvmfd-4026531836-4049952@", "flatshare-engine.service", env=marked)
    host.socket(f"{socket_dir}/engine.sock", "flatshare-engine.service", env=marked)
    done = host.audit("--expect", f"{socket_dir}/engine.sock")
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        f"audit-listeners: ok: the box's units hold 1 Unix sockets, all in {host.runtime}, "
        "and 0 TCP or UDP listeners, all allowed; the GPU servers run in the host's network namespace "
        "(FLATSHARE_PRIVATE_NET=host) with 2 loopback TCP listeners and 1 abstract sockets\n"
    )
    # Unmarked, the same listeners are refused. Marked, one off loopback is, and so are a loopback port outside the
    # ephemeral range (a KV-event publisher's 5557) and an abstract socket that is not CUDA's (a ZMQ endpoint's),
    # which the namespace would hide and the host's does not; the gateway is no GPU server.
    host = Listeners(tmp_path / "unmarked")
    host.tcp("127.0.1.1:36087", "flatshare-engine.service")
    host.socket("@cuda-uvmfd-1", "flatshare-stt.service")
    host.tcp("0.0.0.0:29500", "flatshare-engine.service", env=marked)
    host.tcp("127.0.0.1:9000", "flatshare-gateway.service", "flatshare-gateway", env=marked)
    host.tcp("127.0.0.1:5557", "flatshare-engine.service", env=marked)
    host.socket("@flatshare-zmq", "flatshare-engine.service", env=marked)
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        "audit-listeners: flatshare-engine.service listens on tcp 127.0.1.1:36087",
        "audit-listeners: flatshare-engine.service listens on tcp 0.0.0.0:29500",
        "audit-listeners: flatshare-gateway.service listens on tcp 127.0.0.1:9000",
        "audit-listeners: flatshare-engine.service listens on tcp 127.0.0.1:5557",
        f"audit-listeners: flatshare-stt.service listens on an abstract socket outside {host.runtime}: "
        "@cuda-uvmfd-1",
        f"audit-listeners: flatshare-engine.service listens on an abstract socket outside {host.runtime}: "
        "@flatshare-zmq",
    ]


def test_the_audit_judges_nothing_from_another_network_namespace_than_the_user_managers(
    tmp_path: Path,
) -> None:
    # ss lists the audit's own namespace: run inside a server's (the RUNBOOK's nsenter line, adapted), it would take
    # that namespace's listeners for the host's and meet --expect through the namespaces' tables, passing while it
    # checked nothing. The host's namespace is the user manager's, and anywhere else the audit exits 2.
    host = Listeners(tmp_path)
    host.socket(f"{host.runtime}/flatshare/gateway.sock", "flatshare-gateway.service")
    host.manager()
    (host.proc / "self" / "ns" / "net").unlink()
    (host.proc / "self" / "ns" / "net").symlink_to("net:[4026532864]")
    done = host.audit()
    assert done.returncode == 2 and done.stdout == ""
    assert done.stderr == (
        f"audit-listeners: run it in the host's network namespace ({HOST_NS}, the user manager's): this one is "
        "net:[4026532864]\n"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every namespace")
def test_the_audit_judges_from_its_own_network_namespace_where_the_managers_is_not_readable(
    tmp_path: Path,
) -> None:
    # The user manager's namespace is the host's, but the kernel may keep it from the user: a manager it started
    # under another identity is not dumpable, as GitHub's Ubuntu 24.04 runners and the A100's Ubuntu 24.04 have it.
    # The audit then judges from its own namespace, as where no manager runs, instead of judging nothing.
    host = Listeners(tmp_path)
    host.socket(f"{host.runtime}/flatshare/gateway.sock", "flatshare-gateway.service")
    [manager] = re.findall(r"pid=(\d+)", host.manager())
    (host.proc / manager / "ns").chmod(0)
    try:
        done = host.audit()
    finally:
        (host.proc / manager / "ns").chmod(0o755)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("audit-listeners: ok: the box's units hold 1 Unix sockets")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every namespace")
def test_the_audit_fails_on_a_process_of_ours_whose_network_namespace_it_cannot_read(
    tmp_path: Path,
) -> None:
    # A process of ours whose namespace the audit may not read listens where the audit cannot say: a failure, not a
    # process skipped without a word.
    host = Listeners(tmp_path)
    engine = [f"{host.runtime}/flatshare/engine.sock"]
    pid = host.namespaced("flatshare-engine.service", "net:[4026532864]", engine)
    (host.proc / pid / "ns").chmod(0)
    try:
        done = host.audit()
    finally:
        (host.proc / pid / "ns").chmod(0o755)
    assert done.returncode == 1
    assert done.stderr == (
        f"audit-listeners: cannot read the network namespace of flatshare-engine.service ({pid}: Permission denied)\n"
    )


def _private_net_works() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        probe = [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--",
            "sh",
            "-c",
            "ip link set lo up",
        ]
        return subprocess.run(probe, capture_output=True).returncode == 0
    except OSError:
        return False


SHOW_NET = (
    "import os; "
    "print(os.getpid()); "
    "print(os.readlink('/proc/self/ns/net')); "
    "print(' '.join(line.split(':')[0].strip() for line in open('/proc/net/dev').readlines()[2:])); "
    "print(os.environ.get('FLATSHARE_PRIVATE_NET'))"
)


@pytest.mark.skipif(
    not _private_net_works(), reason="needs Linux with unprivileged user namespaces"
)
def test_in_private_net_runs_its_command_alone_on_a_loopback() -> None:
    # The command keeps the wrapper's process (no fork on the way), so it stays the unit's main process: what
    # KillSignal= reaches, wait-healthy --pid watches and the RUNBOOK's nsenter --target finds.
    wrapper = subprocess.Popen(
        ["bash", str(DEPLOY / "bin" / "in-private-net"), sys.executable, "-c", SHOW_NET],
        env={k: v for k, v in os.environ.items() if k != "FLATSHARE_PRIVATE_NET"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = wrapper.communicate(timeout=30)
    assert wrapper.returncode == 0, err
    pid, space, interfaces, marked = out.splitlines()
    assert int(pid) == wrapper.pid
    assert space != os.readlink("/proc/self/ns/net") and interfaces == "lo" and marked == "None"


def _refusing_unshare(tmp_path: Path) -> dict[str, str]:
    """An environment whose unshare fails as a kernel that refuses unprivileged user namespaces makes it fail."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "unshare").write_text(
        "#!/bin/sh\necho 'unshare: unshare failed: Operation not permitted' >&2\nexit 1\n"
    )
    (bin_dir / "unshare").chmod(0o755)
    return {k: v for k, v in os.environ.items() if k != "FLATSHARE_PRIVATE_NET"} | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    }


def test_in_private_net_runs_its_command_in_the_hosts_namespace_where_the_host_is_marked(
    tmp_path: Path,
) -> None:
    # A host whose kernel refuses unprivileged user namespaces, which the preflight found and host.py's drop-ins
    # mark (FLATSHARE_PRIVATE_NET=host): the command runs as it is, without trying unshare, the mark in its environment
    # for the audit, in the wrapper's own process.
    show = [
        sys.executable,
        "-c",
        "import os; print(os.getpid(), os.environ.get('FLATSHARE_PRIVATE_NET'))",
    ]
    wrapper = subprocess.Popen(
        ["bash", str(DEPLOY / "bin" / "in-private-net"), *show],
        env=_refusing_unshare(tmp_path) | {"FLATSHARE_PRIVATE_NET": "host"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = wrapper.communicate(timeout=30)
    assert wrapper.returncode == 0 and out.split() == [str(wrapper.pid), "host"], err
    assert err == ""


def test_in_private_net_fails_where_the_namespace_cannot_be_made_and_the_host_is_not_marked(
    tmp_path: Path,
) -> None:
    # Unmarked, a namespace that cannot be made fails the unit with unshare's own error, and the command does not
    # run: no server starts in the host's namespace unless the host's render says so.
    show = [sys.executable, "-c", "print('ran')"]
    done = subprocess.run(
        ["bash", str(DEPLOY / "bin" / "in-private-net"), *show],
        env=_refusing_unshare(tmp_path),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1 and done.stdout == ""
    assert "unshare: unshare failed: Operation not permitted" in done.stderr


UBUNTU_REFUSES = "unshare: write failed /proc/self/uid_map: Operation not permitted"
"""unshare's error on Ubuntu 24.04, whose kernel.apparmor_restrict_unprivileged_userns=1 refuses an unprivileged user
namespace its uid map (Gray's, 2026-10-27)."""


def namespace_tools(bin_dir: Path, *, refused: bool, unshare: bool = True) -> Path:
    """unshare and ip stand-ins in ``bin_dir``: unshare logs its arguments to ``bin_dir``/unshare.log and runs the
    command after its ``--``, or, ``refused``, fails as Gray's kernel makes it fail; ip succeeds. Without ``unshare``,
    only ip. Returns ``bin_dir``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    tools = {"ip": "#!/bin/sh\nexit 0\n"}
    if unshare:
        run = (
            f"echo '{UBUNTU_REFUSES}' >&2\nexit 1\n"
            if refused
            else ('while [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n')
        )
        tools["unshare"] = f'#!/bin/sh\necho "$*" >>"{bin_dir}/unshare.log"\n{run}'
    for name, text in tools.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    return bin_dir


@pytest.mark.parametrize("refused", [False, True])
def test_private_net_check_tries_what_in_private_net_does(tmp_path: Path, refused: bool) -> None:
    # The check that decides whether a GPU node's servers get network namespaces of their own runs the command that
    # in-private-net runs at every start, and gives unshare's own error where the kernel refuses it.
    bin_dir = namespace_tools(tmp_path / "bin", refused=refused)
    done = subprocess.run(
        ["bash", str(DEPLOY / "bin" / "private-net-check")],
        env=os.environ | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert (
        bin_dir / "unshare.log"
    ).read_text() == "--user --map-root-user --net -- sh -c ip link set lo up\n"
    if refused:
        assert done.returncode == 1
        assert done.stdout.startswith(f"refused: {UBUNTU_REFUSES}")
    else:
        assert (done.returncode, done.stdout) == (0, "allowed\n")
    # in-private-net, unmarked, runs the same command (its script's line), so the check cannot pass where it fails.
    assert (
        "unshare --user --map-root-user --net -- sh -c 'ip link set lo up"
        in (DEPLOY / "bin" / "in-private-net").read_text()
    )


def test_private_net_check_without_unshare_tries_nothing_and_says_it_cannot_tell(
    tmp_path: Path,
) -> None:
    bin_dir = namespace_tools(tmp_path / "bin", refused=True, unshare=False)
    bash = shutil.which("bash")  # found before PATH holds the stand-ins' directory alone
    assert bash is not None
    done = subprocess.run(
        [bash, str(DEPLOY / "bin" / "private-net-check")],
        env=os.environ | {"PATH": str(bin_dir)},
        capture_output=True,
        text=True,
    )
    assert (done.returncode, done.stdout) == (2, "unknown: no unshare on PATH\n")


def test_the_gpu_nodes_servers_run_in_network_namespaces_of_their_own() -> None:
    # Every unit that starts a GPU server, rendered or a template, and every engine variant that replaces the
    # engine's start, starts it through in-private-net; SGLang alone, which serves TCP to the proxy socket in the
    # host's namespace, does not.
    wrapper = "%h/flatshare/src/deploy/bin/in-private-net "
    units = [
        directory / f"{unit}.service"
        for directory in (DEPLOY / "phastform" / "systemd", DEPLOY / "templates" / "systemd")
        for unit in ("flatshare-engine", "flatshare-stt", "flatshare-tts")
    ]
    variants = [
        path
        for directory in (DEPLOY / "phastform" / "variants", DEPLOY / "templates" / "variants")
        for unit in ("flatshare-engine", "flatshare-stt", "flatshare-tts")
        for path in directory.glob(f"*/{unit}.conf")
    ]
    for path in [*units, *variants]:
        starts = [start for start in exec_lines(path.read_text(), "ExecStart") if start]
        if not starts or "sglang" in starts[-1]:
            continue
        assert starts[-1].startswith(wrapper), path


def test_the_gateways_socket_unit_listens_where_gateway_toml_does() -> None:
    systemd = DEPLOY / "cyan" / "systemd"
    socket = (systemd / "flatshare-gateway.socket").read_text()
    listens = {
        line.partition("=")[2] for line in socket.splitlines() if line.startswith("ListenStream=")
    }
    example = tomllib.loads((DEPLOY / "cyan" / "gateway.example.toml").read_text())
    entries = {entry.get("tcp") or entry["unix"] for entry in example["listen"]}
    # The gateway refuses an inherited socket that no [[listen]] entry names; %t is cyan's /run/user/1007.
    assert {listen.replace("%t", "/run/user/1007") for listen in listens} == entries
    service = (systemd / "flatshare-gateway.service").read_text()
    for line in ("Type=notify", "NotifyAccess=all", "Requires=flatshare-gateway.socket"):
        assert line in service.splitlines(), line
    assert "flatshare-gateway.socket" in exec_lines(service, "After")[0].split()
    assert exec_lines(service, "ExecReload") == [
        "%h/flatshare/src/deploy/bin/flatshare-gateway-reload %n $MAINPID"
    ]


GATEWAY = """\
import signal, sys, time
signal.signal(signal.SIGHUP, lambda *_: open(sys.argv[1], "w").write("reload"))
print("up", flush=True)
time.sleep(30)
"""
"""A gateway's main process to the reload helper: it marks a SIGHUP in a file."""


def reload_gateway(work: Path, takes_over: bool) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Runs deploy/bin/flatshare-gateway-reload against a main process that marks SIGHUP, with a systemctl whose MainPID
    changes once the mark is there if ``takes_over``, as when the successor tells systemd it is the main process."""
    marked = work / "sighup"
    gateway = subprocess.Popen(
        [sys.executable, "-c", GATEWAY, str(marked)], stdout=subprocess.PIPE, text=True
    )
    assert gateway.stdout.readline() == "up\n"
    successor = "4242" if takes_over else str(gateway.pid)
    bin_dir = work / "bin"
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(
        "#!/bin/sh\n"
        f'[ "$*" = "--user show --property MainPID --value flatshare-gateway.service" ] || exit 2\n'
        f"if [ -e {marked} ]; then echo {successor}; else echo {gateway.pid}; fi\n"
    )
    (bin_dir / "systemctl").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FLATSHARE_GATEWAY_RELOAD_WAIT_S": "1",
    }
    try:
        done = subprocess.run(
            [
                DEPLOY / "bin" / "flatshare-gateway-reload",
                "flatshare-gateway.service",
                str(gateway.pid),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        gateway.kill()
        gateway.wait()
    return done, marked


def test_a_reload_succeeds_once_the_successor_is_the_main_process(tmp_path: Path) -> None:
    done, marked = reload_gateway(tmp_path, takes_over=True)
    assert marked.read_text() == "reload", "the main process got SIGHUP"
    assert done.returncode == 0, done.stderr
    assert done.stdout.endswith(
        "flatshare-gateway.service's main process is 4242, after " + done.args[2] + "\n"
    )


def test_a_reload_fails_while_the_old_process_is_still_the_main_one(tmp_path: Path) -> None:
    done, marked = reload_gateway(tmp_path, takes_over=False)
    assert marked.exists()
    assert done.returncode == 1
    assert "main process is still" in done.stderr and "after 1 s" in done.stderr


def test_a_stop_during_a_reload_does_not_fail_the_unit(tmp_path: Path) -> None:
    # A stop that arrives while the reload waits for the successor signals the unit's whole cgroup, the script and
    # the command it waits in included, and both gateways drain under systemd's wait for the cgroup: the script exits
    # 0 at the SIGTERM. It died of it before, and the unit ended failed (result 'signal') with both drained.
    marked = tmp_path / "sighup"
    gateway = subprocess.Popen(
        [sys.executable, "-c", GATEWAY, str(marked)], stdout=subprocess.PIPE, text=True
    )
    assert gateway.stdout is not None and gateway.stdout.readline() == "up\n"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(f"#!/bin/sh\necho {gateway.pid}\n")  # no successor yet
    (bin_dir / "systemctl").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FLATSHARE_GATEWAY_RELOAD_WAIT_S": "30",
    }
    reload = subprocess.Popen(
        [
            DEPLOY / "bin" / "flatshare-gateway-reload",
            "flatshare-gateway.service",
            str(gateway.pid),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not marked.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marked.exists(), "the gateway got its SIGHUP"
        time.sleep(0.3)  # the script waits for the successor
        os.killpg(reload.pid, signal.SIGTERM)  # as systemd's stop signals the unit's cgroup
        _, stderr = reload.communicate(timeout=10)
        assert reload.returncode == 0, stderr
    finally:
        reload.kill()
        reload.wait()
        gateway.kill()
        gateway.wait()


DRAINING = """\
import signal, sys, time
def drain(*_):
    time.sleep(0.5)
    open(sys.argv[1], "w").write("drained")
    sys.exit(0)
signal.signal(signal.SIGTERM, drain)
print("up", flush=True)
time.sleep(30)
"""
"""A gateway's main process to the stop helper: on SIGTERM it drains for 0.5 s, marks that in a file and exits 0."""
HUNG = """\
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("up", flush=True)
time.sleep(60)
"""
"""A gateway's main process to the stop helper that never exits on SIGTERM: a drain that hangs."""
needs_pidfds = pytest.mark.skipif(
    not hasattr(os, "pidfd_open"),
    reason="flatshare-gateway-stop holds the process by a pidfd (Linux)",
)


def stop_gateway(
    work: Path, unit: str, gateway_py: str = DRAINING, *options: str, timeout: float = 30
) -> tuple[subprocess.CompletedProcess[str], subprocess.Popen[str], Path, float]:
    """Runs deploy/bin/flatshare-gateway-stop [OPTIONS] UNIT PID against a main process of flatshare-gateway.service, as
    deploy/bin/flatshare-svc runs one (FLATSHARE_SVC_UNIT in its environment): what it did, the process, its mark and
    how long the stop took. A stop that outlasts ``timeout`` raises, the process killed."""
    marked = work / "drained"
    gateway = subprocess.Popen(
        [sys.executable, "-c", gateway_py, str(marked)],
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ | {"FLATSHARE_SVC_UNIT": "flatshare-gateway.service"},
    )
    assert gateway.stdout is not None and gateway.stdout.readline() == "up\n"
    started = time.monotonic()
    try:
        done = subprocess.run(
            [DEPLOY / "bin" / "flatshare-gateway-stop", *options, unit, str(gateway.pid)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        gateway.kill()
        gateway.wait()
        raise
    return done, gateway, marked, time.monotonic() - started


@needs_pidfds
def test_a_stop_returns_once_the_gateway_has_drained_and_exited(tmp_path: Path) -> None:
    # The unit's ExecStop, which systemd waits for: after a reload systemd itself would SIGKILL the successor at once.
    # Its wait, 45 s, is the drain's limit (drain_timeout_s, 30 s) and a margin, inside systemd's 90 s stop timeout.
    service = (DEPLOY / "cyan" / "systemd" / "flatshare-gateway.service").read_text()
    assert exec_lines(service, "ExecStop") == [
        "%h/flatshare/src/deploy/bin/flatshare-gateway-stop --wait 45 %n $MAINPID"
    ]
    done, gateway, marked, took = stop_gateway(tmp_path, "flatshare-gateway.service")
    assert done.returncode == 0, done.stderr
    assert marked.read_text() == "drained" and took >= 0.5
    assert gateway.wait(timeout=5) == 0


@needs_pidfds
def test_a_stop_kills_a_gateway_still_running_at_the_end_of_its_wait(tmp_path: Path) -> None:
    # A gateway that ignores SIGTERM (only a hung one runs on past its drain) gets SIGKILL once the wait is over, so
    # that the stop takes the wait, not twice the unit's TimeoutStopSec (ExecStop's, then the SIGTERM's that
    # follows it); the stop fails, as the drain did. Before, the stop waited for as long as the process ran.
    done, gateway, marked, took = stop_gateway(
        tmp_path, "flatshare-gateway.service", HUNG, "--wait", "1", timeout=10
    )
    try:
        assert done.returncode == 1, done.stderr
        assert "did not exit within 1 s of SIGTERM: sending SIGKILL" in done.stderr
        assert 1 <= took < 3, took
        assert gateway.wait(timeout=5) == -signal.SIGKILL and not marked.exists()
    finally:
        gateway.kill()
        gateway.wait()


def test_a_stops_wait_is_a_number_of_seconds() -> None:
    for options in (["--wait"], ["--wait", "0"], ["--wait", "soon"], ["--wait", "nan"]):
        done = subprocess.run(
            [DEPLOY / "bin" / "flatshare-gateway-stop", *options, "flatshare-gateway.service"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (done.returncode, done.stderr) == (
            2,
            "usage: flatshare-gateway-stop [--wait SECONDS] UNIT [PID]\n",
        ), options
    # Without a main process ($MAINPID empty) there is nothing to wait for, whatever the wait.
    done = subprocess.run(
        [DEPLOY / "bin" / "flatshare-gateway-stop", "--wait", "2.5", "flatshare-gateway.service"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (done.returncode, done.stderr) == (0, "")


@needs_pidfds
def test_a_stop_leaves_alone_a_process_that_is_not_the_units(tmp_path: Path) -> None:
    done, gateway, marked, _ = stop_gateway(tmp_path, "flatshare-smg.service")
    try:
        assert done.returncode == 0
        assert "is not flatshare-smg.service's: nothing to stop" in done.stderr
        assert gateway.poll() is None and not marked.exists()
    finally:
        gateway.kill()
        gateway.wait()
    # Without a main process ($MAINPID empty), or once it is gone, there is nothing to stop either.
    for args in ((), (str(gateway.pid),)):
        done = subprocess.run(
            [DEPLOY / "bin" / "flatshare-gateway-stop", "flatshare-gateway.service", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (done.returncode, done.stderr) == (0, ""), args


def exec_lines(text: str, key: str) -> list[str]:
    """The values of a unit's ``key=`` lines, continued lines joined."""
    lines = text.replace("\\\n", " ").splitlines()
    return [line.partition("=")[2] for line in lines if line.startswith(f"{key}=")]


def test_the_gpu_servers_start_at_once_and_each_variant_orders_only_what_it_needs() -> None:
    # Both vLLM servers fix their KV pools, so neither profiles memory at its start and nothing orders the three
    # GPU servers among themselves: flatshare.target starts them at once. The mps variant orders each after the
    # MPS daemon it needs; SGLang sizes its pool from the memory free at its start, so its engine waits for the
    # audio servers, as every measured start of the variant did.
    servers = {"flatshare-engine.service", "flatshare-stt.service", "flatshare-tts.service"}

    def ordering(text: str) -> set[str]:
        keys = ("After", "Before", "Requires", "BindsTo", "Wants")
        return {name for key in keys for line in exec_lines(text, key) for name in line.split()}

    for directory in (DEPLOY / "phastform" / "systemd", DEPLOY / "templates" / "systemd"):
        target = (directory / "flatshare.target").read_text()
        assert ordering(target) == servers, directory
        for unit in servers:
            assert not ordering((directory / unit).read_text()) & servers, (directory, unit)
    for unit in servers:
        conf = (
            DEPLOY / "phastform" / "variants" / "mps" / unit.replace(".service", ".conf")
        ).read_text()
        assert (
            exec_lines(conf, "Requires") == exec_lines(conf, "After") == ["flatshare-mps.service"]
        ), unit
    sglang = DEPLOY / "phastform" / "variants" / "sglang" / "flatshare-engine.conf"
    for conf in [
        *(DEPLOY / "phastform" / "variants").glob("*/*.conf"),
        *(DEPLOY / "templates" / "variants").glob("*/*.conf"),
    ]:
        among = ordering(conf.read_text()) & servers
        assert among == (
            {"flatshare-stt.service", "flatshare-tts.service"} if conf == sglang else set()
        ), conf


def test_the_gpu_nodes_servers_listen_on_their_sockets_in_a_private_directory() -> None:
    phastform = DEPLOY / "phastform"
    for config in ("engine.yaml", "stt.yaml"):
        keys = re.findall(r"^([a-z-]+):", (phastform / config).read_text(), re.M)
        assert "host" not in keys and "port" not in keys, config
    sockets = {
        "flatshare-engine": "${FLATSHARE_ENGINE_SOCKET}",
        "flatshare-stt": "${FLATSHARE_STT_SOCKET}",
        "flatshare-tts": "%t/flatshare/tts.sock",
    }
    for unit, socket in sockets.items():
        text = (phastform / "systemd" / f"{unit}.service").read_text()
        [start] = exec_lines(text, "ExecStart")
        assert f"--uds {socket}" in start and "--port" not in start and "--host" not in start, unit
        pre = exec_lines(text, "ExecStartPre")
        assert pre[0] == "/usr/bin/install -d -m 0700 %t/flatshare", unit
        if (
            unit != "flatshare-tts"
        ):  # vLLM leaves a killed server's socket file; uvicorn replaces it itself
            assert f"/usr/bin/rm -f {socket}" in pre, unit
        [ready] = exec_lines(text, "ExecStartPost")
        assert ready.startswith(
            f"%h/flatshare/src/deploy/bin/wait-healthy --unix-socket {socket} --pid=${{MAINPID}} "
        ), unit
    engine = (phastform / "systemd" / "flatshare-engine.service").read_text()
    assert "Environment=FLATSHARE_ENGINE_SOCKET=%t/flatshare/engine.sock" in engine.splitlines()
    # Every variant that starts its own vLLM serves the same socket; SGLang, which cannot, gets a proxy to its port.
    for conf in (phastform / "variants").glob("*/flatshare-engine.conf"):
        starts = exec_lines(conf.read_text(), "ExecStart")
        if not starts:
            continue
        if "sglang" in starts[-1]:
            assert "Wants=flatshare-engine-proxy.socket" in conf.read_text().splitlines(), conf
        else:
            assert "--uds ${FLATSHARE_ENGINE_SOCKET} $FLATSHARE_ENGINE_FLAGS" in starts[-1], conf
    proxy = (phastform / "systemd" / "flatshare-engine-proxy.socket").read_text().splitlines()
    assert (
        "ListenStream=%t/flatshare/engine.sock" in proxy
        and "BindsTo=flatshare-engine.service" in proxy
    )
    smoke = (phastform / "smoke.sh").read_text()
    for name in ("engine", "stt", "tts"):
        assert f"${{socket_dir}}/{name}.sock" in smoke or f"$socket_dir/{name}.sock" in smoke, name
    assert '"$(dirname "$0")/../bin/audit-listeners"' in smoke


def test_smgs_arm_listens_on_no_tcp_port_of_cyans_network() -> None:
    # SMG takes only http:// workers and serves TCP only: it runs in the voice-plane namespace, between a proxy
    # from the gateway's socket and one to the engine's tunnel socket.
    unit = (DEPLOY / "cyan" / "systemd" / "flatshare-smg.service").read_text()
    assert exec_lines(unit, "ExecStart") == [
        "%h/flatshare/src/deploy/bin/in-voicenet %h/flatshare/src/deploy/cyan/smg.sh"
    ]
    assert "BindsTo=flatshare-voicenet.service" in unit.splitlines()
    smg = (DEPLOY / "cyan" / "smg.sh").read_text()
    assert '-l 127.0.0.1:18000 "$proxy" "$socket_dir/phastform-engine.sock"' in smg
    assert '-l "$socket_dir/smg.sock" "$proxy" 127.0.0.1:18090' in smg
    assert "--worker-urls http://127.0.0.1:18000" in smg and "--prometheus-host 127.0.0.1" in smg
    arms = [arm for arm in (REPO / "benchmarks" / "e2" / "arms").glob("*.toml")]
    smg_arms = [arm for arm in arms if 'router = "smg"' in arm.read_text()]
    assert smg_arms
    for arm in smg_arms:
        assert 'upstream = "unix:/run/user/1007/flatshare/smg.sock"' in arm.read_text(), arm.name


def test_the_gpu_nodes_servers_keep_idle_connections_for_60_s() -> None:
    # Through the tunnel each new connection is a new SSH session, so the servers keep theirs between turns.
    systemd = DEPLOY / "phastform" / "systemd"
    for unit in ("flatshare-engine", "flatshare-stt"):
        text = (systemd / f"{unit}.service").read_text()
        assert "Environment=VLLM_HTTP_TIMEOUT_KEEP_ALIVE=60" in text.splitlines(), unit
    [start] = exec_lines((systemd / "flatshare-tts.service").read_text(), "ExecStart")
    assert "--timeout-keep-alive 60" in start


FAKES = Path(__file__).parent / "fakes" / "campaign"
"""Stand-ins for ssh and rsync between directories, one per host (the campaign tests' own)."""


def git(repo: Path, *args: str) -> str:
    command = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


STUB_RELEASE = """\
#!/usr/bin/env python3
# flatshare-release on a host in the deploy.sh tests: tree-hash is the real tool's; every other command is logged with
# the files of the upload it names, and status --json answers $HOME/status.json.
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "tree-hash":
    sys.exit(subprocess.run([sys.executable, os.environ["REAL_RELEASE"], *args]).returncode)
if args[:2] == ["status", "--json"]:
    status = Path(os.environ["HOME"]) / "status.json"
    print(status.read_text() if status.exists() else json.dumps({"releases": []}))
    sys.exit(0)
entry = {"args": args}
if args[0] == "deploy":
    upload = Path(args[1])
    entry["files"] = sorted(str(p.relative_to(upload)) for p in upload.rglob("*") if p.is_file())
with open(Path(os.environ["HOME"]) / "release.log", "a") as log:
    log.write(json.dumps(entry) + "\\n")
"""


def deploy_repo(tmp_path: Path, host_dir: str = "phastform") -> tuple[Path, Path, dict[str, str]]:
    """A repository with deploy.sh and a stand-in flatshare-release, committed, and the hosts it deploys to."""
    repo, hosts = tmp_path / "repo", tmp_path / "hosts"
    (repo / "deploy" / host_dir / "systemd").mkdir(parents=True)
    (repo / "deploy" / "bin").mkdir()
    (repo / "deploy" / "deploy.sh").write_text((DEPLOY / "deploy.sh").read_text())
    (repo / "deploy" / "bin" / "flatshare-release").write_text(STUB_RELEASE)
    (repo / "deploy" / host_dir / "systemd" / "flatshare.target").write_text("[Unit]\n")
    (repo / "crates").mkdir()
    (repo / "crates" / "lib.rs").write_text("// the gateway\n")
    (repo / ".gitignore").write_text("__pycache__/\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "first")
    hosts.mkdir()
    env = os.environ | {
        "PATH": f"{FAKES}{os.pathsep}{os.environ['PATH']}",
        "FAKE_HOSTS": str(hosts),
        "FLATSHARE_SSH": str(FAKES / "ssh"),
        "REAL_RELEASE": str(DEPLOY / "bin" / "flatshare-release"),
    }
    return repo, hosts, env


def release_log(home: Path) -> list[dict]:
    path = home / "release.log"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def tree_hash(directory: Path) -> str:
    done = subprocess.run(
        [sys.executable, str(DEPLOY / "bin" / "flatshare-release"), "tree-hash", str(directory)],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def test_deploy_uploads_the_commit_alone_and_runs_its_release_there(tmp_path: Path) -> None:
    repo, hosts, env = deploy_repo(tmp_path)
    (repo / "deploy" / "__pycache__").mkdir()  # ignored: never released
    (repo / "deploy" / "__pycache__" / "x.pyc").write_bytes(b"")
    deploy = ["bash", str(repo / "deploy" / "deploy.sh"), "phastform.example.org"]
    done = subprocess.run(
        [*deploy, "--adopt", "--drain-timeout", "60"], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    home = hosts / "phastform.example.org"
    rev, commit = git(repo, "describe", "--always"), git(repo, "rev-parse", "HEAD")
    rid = git(repo, "rev-parse", "--short=12", "HEAD")
    upload = f"flatshare/releases/.upload-{rid}"
    [entry] = release_log(home)
    # The commit's files and DEPLOYED, uploaded, and the uploaded tool run on them, with the tree's hash.
    assert entry["files"] == sorted([*git(repo, "ls-files").splitlines(), "DEPLOYED"])
    assert (home / upload / "DEPLOYED").read_text() == rev + "\n"
    assert entry["args"] == [
        "deploy", upload, "--host", "phastform", "--id", rid, "--commit", commit, "--describe", rev,
        "--tree", tree_hash(home / upload), "--adopt", "--drain-timeout", "60",
    ]  # fmt: skip
    # A change the release would not carry, committed or not, is refused before anything moves.
    (repo / "deploy" / "deploy.sh").write_text(
        (repo / "deploy" / "deploy.sh").read_text() + "# changed\n"
    )
    (repo / "new.txt").write_text("")
    done = subprocess.run(deploy, env=env, capture_output=True, text=True)
    assert done.returncode == 2
    assert "changes the release would not carry" in done.stderr
    assert " M deploy/deploy.sh" in done.stderr and "?? new.txt" in done.stderr
    assert len(release_log(home)) == 1


def test_deploy_rolls_back_and_shows_status_with_the_current_releases_tool(tmp_path: Path) -> None:
    repo, hosts, env = deploy_repo(tmp_path)
    home = hosts / "phastform"
    tool = home / "flatshare" / "src" / "deploy" / "bin" / "flatshare-release"
    tool.parent.mkdir(parents=True)
    tool.write_text(STUB_RELEASE)
    deploy = ["bash", str(repo / "deploy" / "deploy.sh"), "phastform"]
    done = subprocess.run(
        [*deploy, "--rollback", "--to", "a" * 12, "--settle", "5"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert release_log(home) == [{"args": ["rollback", "--to", "a" * 12, "--settle", "5"]}]
    done = subprocess.run([*deploy, "--status"], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert release_log(home)[-1] == {"args": ["status"]}


def test_deploy_runs_the_newest_releases_tool_while_the_current_release_has_none(
    tmp_path: Path,
) -> None:
    # A rollback to the release adopted from before releases leaves current without a flatshare-release of its own:
    # the newest release's, by its commit's time, shows the status and goes forward again.
    repo, hosts, env = deploy_repo(tmp_path)
    home = hosts / "phastform"
    flatshare = home / "flatshare"
    (flatshare / "releases" / "legacy-v1.5" / "deploy").mkdir(parents=True)
    (flatshare / "current").symlink_to("releases/legacy-v1.5")
    (flatshare / "src").symlink_to("current")
    stub = STUB_RELEASE.replace(
        'entry = {"args": args}', 'entry = {"args": args, "tool": sys.argv[0]}'
    )
    now = time.time()
    for rid, age in (("a" * 12, 100), ("b" * 12, 10), ("c" * 12, 1000)):
        tool = flatshare / "releases" / rid / "deploy" / "bin" / "flatshare-release"
        tool.parent.mkdir(parents=True)
        tool.write_text(stub)
        os.utime(tool, (now - age, now - age))
    deploy = ["bash", str(repo / "deploy" / "deploy.sh"), "phastform"]
    for args in (["--status"], ["--rollback"]):
        done = subprocess.run([*deploy, *args], env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    newest = f"flatshare/releases/{'b' * 12}/deploy/bin/flatshare-release"
    assert release_log(home) == [
        {"args": ["status"], "tool": newest},
        {"args": ["rollback"], "tool": newest},
    ]


def gateway_build(repo: Path, out: Path, commit: str) -> Path:
    """A gateway binary as deploy/build-gateway.sh leaves one, built from ``commit``."""
    out.write_text("#!/bin/sh\necho 'flatshare-gateway 1.0.0'\n")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / f"{out.name}.json").write_text(json.dumps({"commit": commit, "sha256": sha}))
    return out


def test_deploy_releases_the_gateway_built_from_this_commit_or_one_of_the_same_crates(
    tmp_path: Path,
) -> None:
    repo, hosts, env = deploy_repo(tmp_path, "cyan")
    (repo / "deploy" / "cyan" / "systemd" / "flatshare-gateway.service").write_text("[Service]\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "the gateway's unit")
    home = hosts / "cyan"
    commit = git(repo, "rev-parse", "HEAD")
    deploy = ["bash", str(repo / "deploy" / "deploy.sh"), "cyan"]
    # No binary, and nothing on the host to carry: refused.
    done = subprocess.run(deploy, env=env, capture_output=True, text=True)
    assert (
        done.returncode == 2
        and "build this commit's (deploy/build-gateway.sh BUILD_HOST)" in done.stderr
    )
    # One built from another commit: refused.
    other = gateway_build(repo, tmp_path / "gateway-other", "0" * 40)
    done = subprocess.run(
        [*deploy, "--gateway", str(other)], env=env, capture_output=True, text=True
    )
    assert (
        done.returncode == 2 and f"was built from {'0' * 40}, not from this commit" in done.stderr
    )
    # This commit's: in the release, with its provenance.
    built = gateway_build(repo, tmp_path / "gateway", commit)
    done = subprocess.run(
        [*deploy, "--gateway", str(built)], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    [entry] = release_log(home)
    assert {"bin/flatshare-gateway", "bin/flatshare-gateway.json"} <= set(entry["files"])
    assert "--gateway-from" not in entry["args"]
    # The host runs a release whose binary was built from crates this commit has unchanged: carried from it.
    current = {
        "id": "c" * 12,
        "commit": "c" * 40,
        "current": True,
        "artifacts": {"bin/flatshare-gateway": {"built": {"commit": commit}}},
    }
    (home / "status.json").write_text(json.dumps({"releases": [current]}))
    (repo / "README.md").write_text("docs only\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "docs")
    tool = home / "flatshare" / "src" / "deploy" / "bin" / "flatshare-release"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text(STUB_RELEASE)
    done = subprocess.run(deploy, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    entry = release_log(home)[-1]
    assert entry["args"][-2:] == ["--gateway-from", "c" * 12]
    assert "bin/flatshare-gateway" not in entry["files"]
    # A commit that changes the crates needs its own build.
    (repo / "crates" / "lib.rs").write_text("// changed\n")
    git(repo, "commit", "-q", "-am", "the gateway changes")
    done = subprocess.run(deploy, env=env, capture_output=True, text=True)
    assert done.returncode == 2 and "has no binary built from these crates" in done.stderr
