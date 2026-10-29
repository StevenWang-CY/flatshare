"""deploy/bin/flatshare-release on a temporary host: a release prepared from an upload (its environments built or
linked by key, its manifest, read-only and renamed into place), the editable installs pinned to the release a process
tree started under, verification before any switch, a switch that restarts or reloads only what changed, the automatic
switch back when a unit or a check fails after the switch, rollback, the drain, pruning, and the adoption of a host
deployed before releases; and that the real hosts' tables name every path their units read. The user manager and uv
are stand-ins (state in files, every call logged); the environments are real venvs of the test's own Python."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
from servers import UnixServer, serve, stop

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
TOOL = DEPLOY / "bin" / "flatshare-release"

loader = importlib.machinery.SourceFileLoader("flatshare_release", str(TOOL))
spec = importlib.util.spec_from_loader("flatshare_release", loader)
release = importlib.util.module_from_spec(spec)
sys.modules["flatshare_release"] = release  # its dataclasses look their module up
loader.exec_module(release)

PY = f"{sys.version_info.major}.{sys.version_info.minor}"

FAKE_SYSTEMCTL = """\
#!{python}
# systemctl --user ...: the units' states in $FAKE_SYSTEMD/state.json, every call logged to calls.log. A unit restarted
# while current names a release listed in $FAKE_SYSTEMD/failing fails; with restart-fails there, so does the command.
import json, os, sys
from pathlib import Path
base = Path(os.environ["FAKE_SYSTEMD"])
path = base / "state.json"
state = json.loads(path.read_text()) if path.exists() else {{}}
args = [a for a in sys.argv[1:] if a != "--user"]
with (base / "calls.log").open("a") as log:
    log.write(" ".join(args) + "\\n")
command, rest = args[0], args[1:]
units = [a for a in rest if not a.startswith("-")]
try:
    current = Path(os.readlink(Path(os.environ["HOME"]) / "flatshare" / "current")).name
except OSError:
    current = None
failing = set((base / "failing").read_text().split()) if (base / "failing").exists() else set()
if command == "is-active":
    sys.exit(0 if all(state.get(u, {{}}).get("ActiveState") == "active" for u in units) else 3)
if command == "show":
    unit = units[0]
    known = state.get(unit, {{}})
    for prop in [a.split("=", 1)[1] for a in rest if a.startswith("--property=")]:
        print(f"{{prop}}={{known.get(prop, '0' if prop in ('NRestarts', 'MainPID') else '')}}")
    sys.exit(0)
if command in ("restart", "reload"):
    for unit in units:
        known = state.setdefault(unit, {{"NRestarts": "0", "MainPID": "0"}})
        known["release"] = current
        known["ActiveState"] = "failed" if current in failing else "active"
        known["MainPID"] = str(int(known.get("MainPID") or 0) + 1)
    path.write_text(json.dumps(state))
    sys.exit(1 if current in failing and (base / "restart-fails").exists() else 0)
sys.exit(0)
"""

FAKE_UV = """\
#!{python}
# uv: venv makes a real venv of this Python; pip sync and install record what they would install; sync makes the project's
# environment with its package and its path dependency (../shared) installed editable, as uv does, by .pth files that name
# their source directories; pip freeze lists a package. A requirements file whose text says FAIL fails the sync.
import json, os, subprocess, sys
from pathlib import Path
with open(os.environ["FAKE_UV_LOG"], "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
args = [a for a in sys.argv[1:] if a != "-q"]
if args[0] == "--version":
    print("uv 0.11.27 (fake)")
    sys.exit(0)
def site(env):
    return Path(env) / "lib" / "python{py}" / "site-packages"
def venv(env):
    if not Path(env, "bin", "python").exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", env], check=True)
if args[0] == "venv":
    venv(args[-1])
elif args[:2] == ["pip", "sync"]:
    if "FAIL" in Path(args[-1]).read_text():
        sys.exit("uv pip sync failed")
elif args[:2] == ["pip", "install"]:
    env = Path(args[args.index("--python") + 1]).parent.parent
    (site(env) / "__editable__.vendor.pth").write_text(args[-1] + "\\n")
elif args[:2] == ["pip", "freeze"]:
    print("fake-package==1.0")
elif args[0] == "sync":
    env, project = os.environ["UV_PROJECT_ENVIRONMENT"], Path(args[args.index("--project") + 1])
    venv(env)
    for package in (project, project.parent / "shared"):
        if (package / "src").is_dir():
            (site(env) / f"_editable_impl_{{package.name}}.pth").write_text(f"{{package / 'src'}}\\n")
            info = site(env) / f"{{package.name}}-1.0.dist-info"
            info.mkdir(exist_ok=True)
            (info / "direct_url.json").write_text(json.dumps({{"url": f"file://{{package}}", "dir_info": {{"editable": True}}}}))
else:
    sys.exit(f"fake uv: {{args}}")
"""

INSTALL = """\
#!/usr/bin/env bash
# The host's install.sh: says which release it ran from.
echo "$(readlink "$HOME/flatshare/current")" >>"$HOME/install.log"
"""

AUDIT = """\
import os, sys
from pathlib import Path
# The listener audit: it fails while current names a release listed in $FAKE_SYSTEMD/audit-fails.
fails = Path(os.environ["FAKE_SYSTEMD"]) / "audit-fails"
current = Path(os.readlink(Path(os.environ["HOME"]) / "flatshare" / "current")).name
failed = fails.exists() and current in fails.read_text().split()
print("audit-listeners: " + ("a unit listens where it may not" if failed else "ok"))
sys.exit(1 if failed else 0)
"""

SPEC = release.HostSpec(
    "testhost",
    node="testhost",
    envs=(
        release.Env(
            "deps",
            inputs=("deps.txt",),
            requirements="deps.txt",
            check="import json; print('deps ok')",
        ),
        release.Env(
            "app",
            inputs=("python/app/uv.lock", "python/app/pyproject.toml"),
            project="python/app",
            args=("--extra", "test"),
            check="import app, shared; print('app', app.VERSION, shared.VERSION)",
        ),
    ),
    units=(
        release.Unit(
            "box-a.service", inputs=("deploy/testhost/a.conf", "deploy/bin/a"), envs=("deps",)
        ),
        release.Unit(
            "box-b.service", inputs=("deploy/testhost/b.conf", "python/app/src/*"), envs=("app",)
        ),
        release.Unit(
            "box-gw.service",
            inputs=("deploy/testhost/gw.service",),
            reload_inputs=("deploy/bin/gw-reload",),
            artifacts=("bin/gw",),
        ),
    ),
    artifacts=("bin/gw",),
)
"""A host of three units: one on a requirements environment, one on a project's, and one with a binary that a reload
takes."""


class Host:
    """A host under ``tmp``: its home with ~/flatshare, its runtime directory, and the stand-ins on PATH."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp
        self.home = tmp / "home"
        self.root = self.home / "flatshare"
        self.root.mkdir(parents=True)
        self.systemd = tmp / "systemd"
        self.systemd.mkdir()
        self.units = self.home / ".config" / "systemd" / "user"
        self.units.mkdir(parents=True)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        for name, text in (("systemctl", FAKE_SYSTEMCTL), ("uv", FAKE_UV)):
            (bin_dir / name).write_text(text.format(python=sys.executable, py=PY))
            (bin_dir / name).chmod(0o755)
        (bin_dir / "loginctl").write_text("#!/bin/sh\n")
        (bin_dir / "loginctl").chmod(0o755)
        self.uv_log = tmp / "uv.log"
        for name, value in {
            "HOME": str(self.home),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_SYSTEMD": str(self.systemd),
            "FAKE_UV_LOG": str(self.uv_log),
            "FLATSHARE_RELEASE_UV": str(bin_dir / "uv"),
            "XDG_RUNTIME_DIR": str(tmp / "run"),
        }.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv(release.PIN, raising=False)
        monkeypatch.setattr(release, "PYTHON", PY)
        (tmp / "run" / "flatshare").mkdir(parents=True)
        self.layout = release.Layout(self.root)

    def upload(self, rid: str, describe: str, **changes: str) -> tuple[Path, str]:
        """The upload of a release: a commit's tree as deploy.sh stages it, with ``changes`` (path: text) over the
        first release's files. Returns it and its tree's hash."""
        files = {
            "DEPLOYED": f"{describe}\n",
            "Cargo.toml": '[workspace.package]\nversion = "9.9.9"\n',
            "deps.txt": "json==1.0\n",
            "python/app/pyproject.toml": '[project]\nname = "app"\n',
            "python/app/uv.lock": "version = 1\n",
            "python/app/src/app/__init__.py": 'VERSION = "v1.0"\n',
            "python/shared/pyproject.toml": '[project]\nname = "shared"\n',
            "python/shared/src/shared/__init__.py": 'VERSION = "v1.0"\n',
            "deploy/testhost/install.sh": INSTALL,
            "deploy/testhost/a.conf": "[Service]\n",
            "deploy/testhost/b.conf": "[Service]\n",
            "deploy/testhost/gw.service": "[Service]\n",
            "deploy/bin/a": "#!/bin/sh\n",
            "deploy/bin/gw-reload": "#!/bin/sh\n",
            "deploy/bin/audit-listeners": AUDIT,
            "bin/gw": "#!/bin/sh\necho 'flatshare-gateway 9.9.9'\n",
        } | changes
        upload = self.root / "releases" / f".upload-{rid}"
        for rel, text in files.items():
            (upload / rel).parent.mkdir(parents=True, exist_ok=True)
            (upload / rel).write_text(text)
        (upload / "bin" / "gw").chmod(0o755)
        sha = hashlib.sha256((upload / "bin" / "gw").read_bytes()).hexdigest()
        (upload / "bin" / "gw.json").write_text(json.dumps({"commit": rid * 3, "sha256": sha}))
        return upload, release.tree_hash(release.tree_files(upload))

    def prepare(self, rid: str, describe: str, **changes: str) -> str:
        upload, tree = self.upload(rid, describe, **changes)
        return release.prepare(
            upload,
            self.layout,
            SPEC,
            rid=rid,
            commit=rid * 3,
            describe=describe,
            tree=tree,
            gateway_from=None,
            systemd=release.Systemd(),
        )

    def switch(self, rid: str, action: str = "switch", **options) -> int:
        settings = {"drain_timeout": 5.0, "force": False, "settle_s": 0.0, "keep": 3} | options
        return release.switch(
            self.layout,
            SPEC,
            rid,
            release.Systemd(),
            action=action,
            units_dir=self.units,
            **settings,
        )

    def start(self, *units: str) -> None:
        """Mark ``units`` running, under whatever release current names."""
        path = self.systemd / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        for unit in units:
            state[unit] = {"ActiveState": "active", "NRestarts": "0", "MainPID": "100"}
        path.write_text(json.dumps(state))

    def state(self) -> dict:
        return json.loads((self.systemd / "state.json").read_text())

    def calls(self) -> list[str]:
        path = self.systemd / "calls.log"
        lines = path.read_text().splitlines() if path.exists() else []
        path.write_text("")
        return [line for line in lines if line.split()[0] in ("restart", "reload")]

    def current(self) -> str | None:
        return self.layout.target(self.layout.current)

    def previous(self) -> str | None:
        return self.layout.target(self.layout.previous)

    def python(self, *code: str, env: dict | None = None) -> str:
        """Runs ``code`` with the app environment's Python, through ~/flatshare/venvs as the units do."""
        done = subprocess.run(
            [str(self.root / "venvs" / "app" / "bin" / "python"), "-c", "\n".join(code)],
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != release.PIN} | (env or {}),
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Host]:
    yield Host(tmp_path, monkeypatch)
    release.writable(tmp_path)  # releases are read-only: pytest removes the tree later


A, B, C, D, E = (c * 12 for c in "abcde")


def test_a_release_is_prepared_from_its_upload_read_only_with_its_manifest(host: Host) -> None:
    rid = host.prepare(A, "v1.0")
    path = host.root / "releases" / rid
    assert not (host.root / "releases" / f".upload-{A}").exists()
    manifest = json.loads((path / "MANIFEST.json").read_text())
    assert (manifest["id"], manifest["commit"], manifest["describe"], manifest["host"]) == (
        A,
        A * 3,
        "v1.0",
        "testhost",
    )
    assert manifest["files"] == release.tree_files(path)
    assert (
        manifest["tree_sha256"]
        == release.tree_hash(manifest["files"])
        == manifest["upload_tree_sha256"]
    )
    assert manifest["files"]["deploy/testhost/a.conf"] == hashlib.sha256(b"[Service]\n").hexdigest()
    assert set(manifest["locks"]) == {"python/app/uv.lock"}
    assert manifest["artifacts"]["bin/gw"]["sha256"] == manifest["files"]["bin/gw"]
    assert manifest["artifacts"]["bin/gw"]["built"]["commit"] == A * 3
    assert set(manifest["units"]) == {"box-a.service", "box-b.service", "box-gw.service"}
    assert set(manifest["units"]["box-gw.service"]) == {"restart", "reload"}
    # Each environment under its key, linked from the release, complete.
    for name in ("deps", "app"):
        directory = manifest["envs"][name]["directory"]
        assert directory == f"{name}-{release.env_key(SPEC.envs[0 if name == 'deps' else 1], path)}"
        assert (path / "venvs" / name).resolve() == (host.root / "envs" / directory).resolve()
        assert (host.root / "envs" / directory / release.MARK).is_file()
    assert manifest["envs"]["app"]["pinned"] == ["python/app/src", "python/shared/src"]
    # Read-only, all of it, and the packages compiled, so that imports need no write.
    for file in (path / "deploy" / "testhost" / "a.conf", path / "python" / "app" / "src"):
        assert not file.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), file
    assert list(
        (path / "python" / "app" / "src" / "app" / "__pycache__").glob(
            f"__init__.cpython-{PY.replace('.', '')}.pyc"
        )
    )
    if os.geteuid() != 0:  # root writes a read-only file all the same
        with pytest.raises(PermissionError):
            (path / "deploy" / "testhost" / "a.conf").write_text("changed")


def test_an_upload_that_is_not_the_staged_tree_is_refused(host: Host) -> None:
    upload, tree = host.upload(A, "v1.0")
    (upload / "deploy" / "testhost" / "a.conf").write_text("[Service]\n# cut short\n")
    with pytest.raises(release.NotReady, match=r"is not the tree deploy\.sh staged"):
        release.prepare(
            upload,
            host.layout,
            SPEC,
            rid=A,
            commit=A * 3,
            describe="v1.0",
            tree=tree,
            gateway_from=None,
            systemd=release.Systemd(),
        )
    assert not (host.root / "releases" / A).exists() and not (host.root / "envs").exists()
    # A binary that is not the one its provenance names.
    upload, tree = host.upload(B, "v1.1", **{"bin/gw": "#!/bin/sh\necho other\n"})
    (upload / "bin" / "gw.json").write_text(json.dumps({"commit": B * 3, "sha256": "0" * 64}))
    tree = release.tree_hash(release.tree_files(upload))
    with pytest.raises(release.NotReady, match="is not the binary its provenance names"):
        release.prepare(
            upload,
            host.layout,
            SPEC,
            rid=B,
            commit=B * 3,
            describe="v1.1",
            tree=tree,
            gateway_from=None,
            systemd=release.Systemd(),
        )


def test_a_release_links_an_unchanged_environment_and_builds_a_changed_one(host: Host) -> None:
    host.prepare(A, "v1.0")
    first = json.loads((host.root / "releases" / A / "MANIFEST.json").read_text())["envs"]
    host.uv_log.write_text("")
    host.prepare(B, "v1.1", **{"python/app/uv.lock": "version = 2\n"})
    second = json.loads((host.root / "releases" / B / "MANIFEST.json").read_text())["envs"]
    assert second["deps"]["directory"] == first["deps"]["directory"]  # linked, not rebuilt
    assert second["app"]["directory"] != first["app"]["directory"]
    builds = [line.split()[0:2] for line in host.uv_log.read_text().splitlines()]
    assert ["venv", "-q"] not in builds and ["pip", "sync"] not in builds  # no deps build
    assert builds.count(["sync", "-q"]) == 1  # the app's
    # The same commit again is the release it was.
    upload, tree = host.upload(B, "v1.1", **{"python/app/uv.lock": "version = 2\n"})
    assert (
        release.prepare(
            upload,
            host.layout,
            SPEC,
            rid=B,
            commit=B * 3,
            describe="v1.1",
            tree=tree,
            gateway_from=None,
            systemd=release.Systemd(),
        )
        == B
    )
    assert not upload.exists()


def test_an_interrupted_environment_build_is_removed_and_built_again(host: Host) -> None:
    with pytest.raises(release.ReleaseError, match="uv pip sync failed"):
        host.prepare(A, "v1.0", **{"deps.txt": "FAIL\n"})
    assert not any((host.root / "envs").glob("deps-*"))  # nothing half built is left
    # A build cut short before its mark: the next release removes it and builds it whole.
    key = release.env_key(SPEC.envs[0], host.upload(B, "v1.1")[0])
    (host.root / "envs" / f"deps-{key}" / "bin").mkdir(parents=True)
    host.prepare(B, "v1.1")
    assert (host.root / "envs" / f"deps-{key}" / release.MARK).is_file()


def test_each_process_tree_imports_the_release_current_named_as_it_started(host: Host) -> None:
    host.prepare(A, "v1.0")
    # The app environment shared: only the packages' code differs.
    new_code = {f"python/{p}/src/{p}/__init__.py": 'VERSION = "v1.1"\n' for p in ("app", "shared")}
    host.prepare(B, "v1.1", **new_code)
    assert (
        json.loads((host.root / "releases" / A / "MANIFEST.json").read_text())["envs"]["app"][
            "directory"
        ]
        == json.loads((host.root / "releases" / B / "MANIFEST.json").read_text())["envs"]["app"][
            "directory"
        ]
    )
    assert host.switch(A) == 0
    assert host.python("import app, shared; print(app.VERSION, shared.VERSION)") == "v1.0 v1.0"
    assert host.switch(B) == 0
    assert host.python("import app, shared; print(app.VERSION, shared.VERSION)") == "v1.1 v1.1"
    # A tree started under A keeps A's code after the switch, its children too, whatever current names.
    child = "import subprocess, sys; print(subprocess.run([sys.executable, '-c', 'import app; print(app.VERSION)'], capture_output=True, text=True).stdout.strip())"
    pinned = {release.PIN: str((host.root / "releases" / A).resolve())}
    assert host.python("import app; print(app.VERSION)", env=pinned) == "v1.0"
    assert host.python(child, env=pinned) == "v1.0"
    assert host.python("import os; print(os.environ['FLATSHARE_RELEASE'])") == str(
        (host.root / "releases" / B).resolve()
    )


def test_a_switch_restarts_only_active_units_whose_inputs_changed(host: Host) -> None:
    host.prepare(A, "v1.0")
    assert host.switch(A) == 0
    host.start("box-a.service", "box-b.service", "box-gw.service")
    host.calls()
    # Only box-a's file changed: box-a restarts.
    host.prepare(B, "v1.1", **{"deploy/testhost/a.conf": "[Service]\nNice=5\n"})
    assert host.switch(B) == 0
    assert host.calls() == ["restart box-a.service"]
    # The gateway's binary changed: a reload. The app's code changed: box-b restarts.
    host.prepare(
        C,
        "v1.2",
        **{
            "deploy/testhost/a.conf": "[Service]\nNice=5\n",
            "bin/gw": "#!/bin/sh\necho 'flatshare-gateway 9.9.9'\n# rebuilt\n",
            "python/app/src/app/__init__.py": 'VERSION = "v1.2"\n',
        },
    )
    assert host.switch(C) == 0
    assert host.calls() == ["restart box-b.service", "reload box-gw.service"]
    # Nothing a unit reads changed (DEPLOYED alone, and a file no unit names): nothing restarts.
    host.prepare(
        D,
        "v1.3",
        **{
            "deploy/testhost/a.conf": "[Service]\nNice=5\n",
            "bin/gw": "#!/bin/sh\necho 'flatshare-gateway 9.9.9'\n# rebuilt\n",
            "python/app/src/app/__init__.py": 'VERSION = "v1.2"\n',
            "README.md": "docs\n",
        },
    )
    assert host.switch(D) == 0
    assert host.calls() == []
    assert (host.current(), host.previous()) == (D, C)
    # An inactive unit is never started by a switch.
    state = host.state()
    state["box-a.service"]["ActiveState"] = "inactive"
    (host.systemd / "state.json").write_text(json.dumps(state))
    host.prepare(
        E,
        "v1.4",
        **{
            "deploy/testhost/a.conf": "[Service]\nNice=9\n",
            "bin/gw": "#!/bin/sh\necho 'flatshare-gateway 9.9.9'\n# rebuilt\n",
            "python/app/src/app/__init__.py": 'VERSION = "v1.2"\n',
        },
    )
    assert host.switch(E) == 0 and host.calls() == []
    assert (host.home / "install.log").read_text().split() == [
        f"releases/{rid}" for rid in (A, B, C, D, E)
    ]


def test_the_compatibility_paths_lead_into_current(host: Host) -> None:
    host.prepare(A, "v1.0")
    host.switch(A)
    for path, target in (
        ("src", "current"),
        ("venvs/app", "../current/venvs/app"),
        ("bin/gw", "../current/bin/gw"),
    ):
        assert os.readlink(host.root / path) == target, path
    assert (host.root / "src" / "DEPLOYED").read_text() == "v1.0\n"
    assert (host.root / "bin" / "gw").resolve() == (
        host.root / "releases" / A / "bin" / "gw"
    ).resolve()


def test_a_release_that_fails_verification_is_not_switched_to(host: Host) -> None:
    host.prepare(A, "v1.0")
    host.switch(A)
    host.start("box-a.service", "box-b.service", "box-gw.service")
    host.calls()
    # The app's code does not import (prepared, as it builds; its check runs it): nothing switches or restarts.
    host.prepare(B, "v1.1", **{"python/app/src/app/__init__.py": "raise ImportError('broken')\n"})
    with pytest.raises(release.ReleaseError, match=r"fails its checks \(env app"):
        host.switch(B)
    assert (host.current(), host.calls()) == (A, [])
    # A release changed after it was prepared, and a binary that says another version.
    release.writable(host.root / "releases" / B)
    (host.root / "releases" / B / "deploy" / "testhost" / "a.conf").write_text("edited in place\n")
    host.prepare(C, "v1.2", **{"bin/gw": "#!/bin/sh\necho 'flatshare-gateway 0.1.0'\n"})
    with pytest.raises(release.ReleaseError, match=r"files, env app"):
        host.switch(B)
    with pytest.raises(release.ReleaseError, match=r"bin/gw --version"):
        host.switch(C)
    assert host.current() == A


def test_a_unit_that_fails_after_the_switch_switches_the_host_back(host: Host) -> None:
    host.prepare(A, "v1.0")
    host.switch(A)
    host.start("box-a.service", "box-b.service", "box-gw.service")
    host.prepare(B, "v1.1")
    host.switch(B)
    host.calls()
    host.prepare(C, "v1.2", **{"deploy/testhost/a.conf": "[Service]\nNice=5\n"})
    (host.systemd / "failing").write_text(C)
    assert host.switch(C) == 1
    # Back on B, previous as it was, box-a restarted on C and again on B, and both recorded.
    assert (host.current(), host.previous()) == (B, A)
    assert host.calls() == ["restart box-a.service", "restart box-a.service"]
    assert host.state()["box-a.service"] == host.state()["box-a.service"] | {
        "ActiveState": "active",
        "release": B,
    }
    history = [
        json.loads(line)
        for line in (host.root / "releases" / "history.jsonl").read_text().splitlines()
    ]
    assert [(h["action"], h["source"], h["target"], h["result"][:6]) for h in history[-2:]] == [
        ("switch", B, C, "failed"),
        ("switch-back", C, B, "ok"),
    ]
    # A restart that fails outright, and a health check that fails, switch back too.
    (host.systemd / "restart-fails").touch()
    assert host.switch(C) == 1 and host.current() == B
    (host.systemd / "failing").unlink()
    (host.systemd / "restart-fails").unlink()
    (host.systemd / "audit-fails").write_text(C)
    assert host.switch(C) == 1 and host.current() == B
    (host.systemd / "audit-fails").unlink()
    assert host.switch(C) == 0 and (host.current(), host.previous()) == (C, B)


def test_switching_back_that_fails_too_says_so(host: Host) -> None:
    host.prepare(A, "v1.0")
    host.switch(A)
    host.start("box-a.service")
    host.prepare(B, "v1.1", **{"deploy/testhost/a.conf": "[Service]\nNice=5\n"})
    (host.systemd / "failing").write_text(f"{A} {B}")
    assert host.switch(B) == 3
    assert host.current() == A


def test_rollback_switches_to_previous_by_the_same_steps(
    host: Host, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    host.prepare(A, "v1.0")
    host.switch(A)
    host.start("box-a.service", "box-b.service", "box-gw.service")
    host.prepare(B, "v1.1", **{"python/app/src/app/__init__.py": 'VERSION = "v1.1"\n'})
    host.switch(B)
    host.calls()
    assert host.switch(host.previous(), action="rollback") == 0
    assert (host.current(), host.previous()) == (A, B)
    assert host.calls() == ["restart box-b.service"]
    assert host.python("import app; print(app.VERSION)") == "v1.0"
    # From the command line: back to B.
    monkeypatch.setitem(release.HOSTS, "testhost", SPEC)
    assert release.main(["--root", str(host.root), "rollback", "--settle", "0"]) == 0
    assert (host.current(), host.previous()) == (B, A)
    assert f"the host runs release {B} (v1.1); previous is {A}" in capsys.readouterr().out


def test_a_drain_waits_for_the_voice_workers_calls_and_a_timeout_switches_nothing(
    host: Host,
) -> None:
    # box-b as the voice worker is: its restart cuts calls, so a switch that restarts it drains it first.
    status, drain = release.voice_files()
    voice = release.Unit(
        "box-b.service",
        inputs=("deploy/testhost/b.conf", "python/app/src/*"),
        envs=("app",),
        drain="voice",
    )
    spec = release.HostSpec(
        SPEC.name, SPEC.node, SPEC.envs, (SPEC.units[0], voice, SPEC.units[2]), SPEC.artifacts
    )
    host.prepare(A, "v1.0")
    release.switch(
        host.layout,
        spec,
        A,
        release.Systemd(),
        action="switch",
        drain_timeout=1,
        force=False,
        settle_s=0,
        keep=3,
        units_dir=host.units,
    )
    host.start("box-b.service")
    status.write_text(json.dumps({"mode": "serve", "active_calls": 1}))
    host.prepare(B, "v1.1", **{"deploy/testhost/b.conf": "[Service]\nNice=5\n"})
    host.calls()
    with pytest.raises(
        release.NotReady, match=r"still has 1 call\(s\) after 0.2 s: nothing switched"
    ):
        release.switch(
            host.layout,
            spec,
            B,
            release.Systemd(),
            action="switch",
            drain_timeout=0.2,
            force=False,
            settle_s=0,
            keep=3,
            units_dir=host.units,
        )
    assert host.current() == A and not drain.exists() and host.calls() == []
    # The calls end: the drain holds new ones off until the restart, and then it is gone.
    status.write_text(json.dumps({"mode": "serve", "active_calls": 0}))
    assert (
        release.switch(
            host.layout,
            spec,
            B,
            release.Systemd(),
            action="switch",
            drain_timeout=1,
            force=False,
            settle_s=0,
            keep=3,
            units_dir=host.units,
        )
        == 0
    )
    assert host.current() == B and not drain.exists() and host.calls() == ["restart box-b.service"]
    # Bench mode has no drain: its restart cuts its calls.
    status.write_text(json.dumps({"mode": "bench", "active_calls": 2}))
    assert release.drain_voice(0.1, force=False) is False and not drain.exists()


def test_an_e2_point_on_the_host_holds_a_switch_off(host: Host) -> None:
    host.prepare(A, "v1.0")
    host.switch(A)
    host.start("box-a.service")
    host.prepare(B, "v1.1", **{"deploy/testhost/a.conf": "[Service]\nNice=5\n"})
    lock = host.root / "runs" / ".flatshare-bench-e2.lock"
    lock.parent.mkdir()
    lock.write_text(f"{os.getpid()}\n")
    with pytest.raises(release.NotReady, match=f"an E2 point \\(pid {os.getpid()}\\) measures"):
        host.switch(B)
    assert host.switch(B, force=True) == 0


def test_pruning_keeps_the_newest_current_previous_and_what_processes_run(
    host: Host, tmp_path: Path
) -> None:
    for i, rid in enumerate((A, B, C, D, E)):
        host.prepare(rid, f"v1.{i}", **{"python/app/uv.lock": f"version = {i}\n"})
    host.switch(A, keep=10)
    host.switch(E, keep=10)
    proc = tmp_path / "proc"
    (proc / "4242").mkdir(parents=True)
    (proc / "4242" / "environ").write_bytes(
        f"HOME=/x\0{release.PIN}={host.root}/releases/{B}\0".encode()
    )
    removed = release.prune(host.layout, keep=1, proc=proc)
    assert sorted(host.layout.ids()) == sorted(
        [A, B, E]
    )  # previous, B's running process, the newest and current
    assert C in removed and D in removed
    envs = {path.name for path in (host.root / "envs").iterdir()}
    assert envs == {
        json.loads((host.root / "releases" / rid / "MANIFEST.json").read_text())["envs"][name][
            "directory"
        ]
        for rid in (A, B, E)
        for name in ("deps", "app")
    }


def test_a_host_deployed_before_releases_is_adopted_in_place(host: Host) -> None:
    # The old layout: the tree synced into src, the venvs and the gateway's binary built in place, a unit running.
    src = host.root / "src"
    (src / "deploy" / "testhost").mkdir(parents=True)
    (src / "DEPLOYED").write_text("v0.9\n")
    (src / "deploy" / "testhost" / "install.sh").write_text(INSTALL)
    (src / "deploy" / "bin").mkdir()
    (src / "deploy" / "bin" / "audit-listeners").write_text(AUDIT)
    (src / "Cargo.toml").write_text('[workspace.package]\nversion = "9.8.0"\n')
    for env in ("deps", "app"):
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(host.root / "venvs" / env)],
            check=True,
        )
    (host.root / "venvs" / "app" / "marker").write_text("built in place\n")
    (host.root / "bin").mkdir()
    (host.root / "bin" / "gw").write_text("#!/bin/sh\necho 'flatshare-gateway 9.8.0'\n")
    (host.root / "bin" / "gw").chmod(0o755)
    (host.root / "bin" / "livekit-server").write_text("a host tool\n")
    inode = (host.root / "venvs" / "app" / "marker").stat().st_ino
    rid = release.adopt(host.layout, SPEC)
    assert rid == "legacy-v0.9" and host.current() == rid
    # The old paths lead to the same files, by links into current.
    assert os.readlink(src) == "current" and (src / "DEPLOYED").read_text() == "v0.9\n"
    assert (host.root / "venvs" / "app" / "marker").stat().st_ino == inode
    assert os.readlink(host.root / "venvs" / "app") == "../current/venvs/app"
    assert (host.root / "envs" / "app-legacy" / "marker").is_file()
    assert (host.root / "bin" / "gw").resolve() == (
        host.root / "releases" / rid / "bin" / "gw"
    ).resolve()
    assert (
        host.root / "bin" / "livekit-server"
    ).read_text() == "a host tool\n"  # a host tool stays
    manifest = json.loads((host.root / "releases" / rid / "MANIFEST.json").read_text())
    assert manifest["adopted"] and manifest["units"] is None and manifest["commit"] is None
    # The first release from it restarts every active unit (the adopted one has no unit hashes), and rolls back to it.
    host.start("box-a.service", "box-b.service")
    host.prepare(A, "v1.0")
    assert host.switch(A) == 0
    assert host.calls() == ["restart box-a.service box-b.service"]
    assert host.switch(rid, action="rollback") == 0
    assert host.current() == rid and (host.root / "venvs" / "app" / "marker").is_file()


def test_the_old_layout_is_refused_without_adopt(host: Host) -> None:
    (host.root / "src").mkdir()
    done = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--root",
            str(host.root),
            "deploy",
            str(host.root / "x"),
            "--host",
            "cyan",
            "--id",
            A,
            "--commit",
            A * 3,
            "--describe",
            "v1",
            "--tree",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "is a deployment from before releases: deploy with --adopt" in done.stdout
    assert (host.root / "src").is_dir() and not (host.root / "src").is_symlink()


def test_one_operation_at_a_time(host: Host) -> None:
    with (
        host.layout.lock(),
        pytest.raises(release.NotReady, match="another flatshare-release operation holds"),
        host.layout.lock(),
    ):
        pass


def test_the_tree_hash_is_the_hash_of_sha256sums_lines_for_its_files(tmp_path: Path) -> None:
    # Its files and nothing else: no bytecode, manifest or links; deploy.sh's on the workstation and the host's agree.
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c.txt").write_text("c\n")
    (tmp_path / "d.txt").write_text("d\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"")
    (tmp_path / "MANIFEST.json").write_text("{}")
    (tmp_path / "link").symlink_to("d.txt")
    lines = "".join(
        f"{hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()}  {rel}\n"
        for rel in ("a/b/c.txt", "d.txt")
    )
    if shutil.which("sha256sum"):
        assert (
            subprocess.run(
                ["sha256sum", "a/b/c.txt", "d.txt"], cwd=tmp_path, capture_output=True, text=True
            ).stdout
            == lines
        )
    assert (
        release.tree_hash(release.tree_files(tmp_path))
        == hashlib.sha256(lines.encode()).hexdigest()
    )
    done = subprocess.run(
        [sys.executable, str(TOOL), "tree-hash", str(tmp_path)], capture_output=True, text=True
    )
    assert done.stdout.strip() == hashlib.sha256(lines.encode()).hexdigest()


def test_the_vendored_checkout_must_be_at_its_pin(host: Host, tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git = ["git", "-C", str(upstream), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    (upstream / "a").write_text("1\n")
    subprocess.run([*git, "add", "a"], check=True)
    subprocess.run([*git, "commit", "-qm", "one"], check=True)
    first = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (upstream / "a").write_text("2\n")
    subprocess.run([*git, "commit", "-qam", "two"], check=True)
    checkout = release.vendor_checkout(host.root, ("vendor/up", str(upstream), first))
    assert (checkout / "a").read_text() == "1\n"
    second = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    with pytest.raises(release.ReleaseError, match=r"bootstrap\.sh moves the checkout"):
        release.vendor_checkout(host.root, ("vendor/up", str(upstream), second))


# The real hosts' tables.

UNIT_PATHS = re.compile(r"%h/flatshare/(src|venvs|bin)/([^\s:'\"]+)")


def unit_files(spec: release.HostSpec, unit: release.Unit) -> list[Path]:
    """``unit``'s file (a template's for an instance, flatshare-tunnel@.service) and, for a service, every drop-in the
    repository can link into it: the variants' and the measurement drop-in (UNIT.conf, linked into UNIT.service.d), and
    the node's own (deploy/NODE/systemd)."""
    file_name = re.sub(r"@[^.]*\.", "@.", unit.name)
    found = [*DEPLOY.glob(f"*/systemd/{file_name}")]
    if file_name.endswith(".service"):
        stem = file_name.removesuffix(".service")
        found += [*DEPLOY.glob(f"*/variants/*/{stem}.conf"), *DEPLOY.glob(f"*/measure/{stem}.conf")]
        if spec.node != "phastform":
            found += [*DEPLOY.glob(f"{spec.node}/systemd/{stem}.conf")]
    return found


@pytest.mark.parametrize("name", sorted(release.HOSTS))
def test_each_hosts_units_name_only_paths_their_inputs_cover(name: str) -> None:
    # A unit that read a release file its inputs did not cover would keep running the old one after a switch that
    # changed it: every %h/flatshare/src path its unit file and drop-ins name is one of its inputs, every venv one of its
    # environments (the sglang variant's aside, which variants/sglang/install.sh builds outside releases), every
    # binary one of its artifacts or the host's tools.
    spec = release.HOSTS[name]
    envs = {env.name for env in spec.envs}
    for unit in spec.units:
        files = unit_files(spec, unit)
        assert any(path.parent.name == "systemd" for path in files), unit.name
        for path in files:
            for kind, rel in UNIT_PATHS.findall(path.read_text()):
                if kind == "src":
                    assert release.matches(rel, unit.inputs + unit.reload_inputs), (unit.name, rel)
                elif kind == "venvs":
                    env = rel.split("/")[0]
                    assert env in unit.envs or env == "sglang", (unit.name, rel)
                    assert env in envs or env == "sglang"
                else:
                    assert f"bin/{rel}" in unit.artifacts + unit.tools, (unit.name, rel)


def test_the_environments_inputs_are_what_decides_them() -> None:
    # A project's environment holds its package and its path dependencies installed editable, whose metadata are their
    # pyproject.toml files: each is an input, as the lock is.
    import tomllib

    for spec_ in release.HOSTS.values():
        for env in spec_.envs:
            for rel in env.inputs:
                assert (REPO / rel).is_file(), rel
            if env.project:
                pyproject = tomllib.loads((REPO / env.project / "pyproject.toml").read_text())
                sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
                paths = {
                    os.path.normpath(f"{env.project}/{s['path']}")
                    for s in sources.values()
                    if "path" in s
                }
                assert set(env.inputs) == {
                    f"{env.project}/uv.lock",
                    f"{env.project}/pyproject.toml",
                } | {f"{p}/pyproject.toml" for p in paths}
            if env.requirements:
                assert env.inputs == (env.requirements,)


def test_bootstrap_moves_the_vendored_checkout_to_the_releases_pin() -> None:
    text = (DEPLOY / "phastform" / "bootstrap.sh").read_text()
    [pin] = re.findall(r"checkout -q ([0-9a-f]{40})", text)
    assert pin == release.TTS.vendor[2]
    assert release.HOSTS["gray"].envs == release.HOSTS["phastform"].envs


def test_the_manifest_records_the_hosts_tools_and_each_environments_interpreter(host: Host) -> None:
    tool = host.root / "bin" / "livekit-server"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\necho 'livekit-server version 1.13.10'\n")
    tool.chmod(0o755)
    spec_ = release.HostSpec(
        SPEC.name,
        SPEC.node,
        SPEC.envs,
        SPEC.units,
        SPEC.artifacts,
        {"bin/livekit-server": "1.13.10"},
    )
    upload, tree = host.upload(A, "v1.0")
    release.prepare(
        upload,
        host.layout,
        spec_,
        rid=A,
        commit=A * 3,
        describe="v1.0",
        tree=tree,
        gateway_from=None,
        systemd=release.Systemd(),
    )
    manifest = json.loads((host.root / "releases" / A / "MANIFEST.json").read_text())
    assert manifest["tools"] == {
        "bin/livekit-server": {
            "sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
            "version": "livekit-server version 1.13.10",
        }
    }
    assert manifest["envs"]["app"]["python"] == ".".join(map(str, sys.version_info[:3]))


def test_a_release_prints_the_serving_profiles_configuration() -> None:
    serving = json.loads((DEPLOY / "phastform" / "serving.json").read_text())
    assert release.describe_serving(serving) == [
        "serving profile rtx-5060-ti-serve (variant make-room-serve), vLLM 0.32.0",
        "  model Qwen/Qwen3-4B-Instruct-2507-FP8@8591804019c8, tokenizer Qwen/Qwen3-4B-Instruct-2507-FP8@8591804019c8, "
        "context 32768",
        "  priority, step cap 256/0, pool 36400 tokens (2275 blocks), CPU tier 20 GiB",
        "  make_room: max_preemptions_per_step 2, budget_cap true, budget_cap_tokens 48, reserve_tokens 0",
        "  scheduler flatshare_make_room_serve.AsyncScheduler over flatshare_make_room_scheduler 0d24d178977f3c1b",
        "  voice: serve mode, 2 calls at once, microphone audio",
        "  gateway: at most 4 agents' sessions beside a node's calls",
    ]


def test_a_serving_hosts_release_holds_the_gateways_nodes_to_the_envelope(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The gateway's GET /nodes on its socket, as the file says; an empty file is a gateway of one upstream (404).
    (short_dir / "flatshare").mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_dir))
    answer = short_dir / "answer"

    class Nodes(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = answer.read_bytes()
            self.send_response(200 if body else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    serving, check = DEPLOY / "phastform" / "serving.json", DEPLOY / "bin" / "serving-check"
    both = ["realtime", "background"]
    view = {
        "nodes": [
            {"name": "phastform", "roles": both, "max_calls": 2, "max_agents": 4},
            {"name": "gray", "roles": both, "max_calls": 2, "max_agents": 12},
        ],
        "sessions": 0,
        "placement": {"calls": "fill", "failover_background_max_in_flight": 0},
    }
    what = "the gateway's nodes against the serving profile's envelope"
    server = serve(UnixServer(str(short_dir / "flatshare" / "gateway.sock"), Nodes))
    try:
        answer.write_text(json.dumps(view))
        assert release.gateway_nodes(serving, check)[:2] == (what, True)
        # A hand-written gateway.toml that runs the calls' node past the envelope fails the release, naming it.
        view["nodes"][0]["max_agents"] = 12
        answer.write_text(json.dumps(view))
        _, ok, detail = release.gateway_nodes(serving, check)
        assert not ok and detail.endswith("nodes.phastform.max_agents"), detail
        answer.write_text("")
        assert release.gateway_nodes(serving, check) == (what, True, "one upstream")
        assert "the gateway runs one upstream" in capsys.readouterr().out
    finally:
        stop(server)
    # No gateway on the socket.
    _, ok, detail = release.gateway_nodes(serving, check)
    assert not ok and detail.startswith("no answer"), detail


def test_the_engine_drain_waits_for_its_requests_and_gives_up_at_its_bound(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The engine's /metrics on its socket: requests in flight while the file says so.
    (short_dir / "flatshare").mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_dir))
    left = short_dir / "left"
    left.write_text("2")

    class Metrics(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            n = float(left.read_text())
            body = (
                f'vllm:num_requests_running{{engine="0",model_name="m"}} {n}\n'
                'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = serve(UnixServer(str(short_dir / "flatshare" / "engine.sock"), Metrics))
    try:
        release.drain_engine(0.5)
        assert (
            "still has 2 request(s) in flight after 0.5 s: its restart cuts them"
            in capsys.readouterr().out
        )
        left.write_text("0")
        release.drain_engine(5)
        assert "the engine has no request in flight" in capsys.readouterr().out
    finally:
        stop(server)
    # No engine on the socket: nothing to wait for.
    release.drain_engine(5)
    assert release.ENGINE_DRAIN_S == 60


def test_scripts_run_from_a_releases_own_directory_link_through_src(tmp_path: Path) -> None:
    # A link into releases/ID dangles once ID is pruned: install.sh and variant.sh, run from there, link through
    # ~/flatshare/src instead, as they do when run through it.
    home = tmp_path / "home"
    tree = home / "flatshare" / "releases" / A
    for node in ("phastform", "cyan", "gray"):
        (tree / "deploy" / node).mkdir(parents=True)
        (tree / "deploy" / node / "install.sh").write_text(
            (DEPLOY / node / "install.sh").read_text()
        )
    (tree / "deploy" / "phastform" / "variant.sh").write_text(
        (DEPLOY / "phastform" / "variant.sh").read_text()
    )
    for rel in (
        "phastform/systemd/flatshare-engine.service",
        "cyan/systemd/flatshare-gateway.service",
        "gray/systemd/flatshare-engine.conf",
    ):
        (tree / "deploy" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / "deploy" / rel).write_text("[Service]\n")
    (tree / "deploy" / "phastform" / "measure").mkdir()
    (tree / "deploy" / "phastform" / "measure" / "flatshare-engine.conf").write_text("[Service]\n")
    (tree / "Cargo.toml").write_text('[workspace.package]\nversion = "1.0.0"\n')
    (home / "flatshare" / "src").symlink_to(f"releases/{A}")  # current's, as on a host
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("systemctl", "loginctl"):
        (bin_dir / tool).write_text('#!/bin/sh\n[ "$2" = is-active ] && exit 3\nexit 0\n')
        (bin_dir / tool).chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    units = home / ".config" / "systemd" / "user"
    src = home / "flatshare" / "src" / "deploy"
    for node in ("phastform", "cyan", "gray"):
        done = subprocess.run(
            ["bash", str(tree / "deploy" / node / "install.sh")],
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
    assert (
        os.readlink(units / "flatshare-engine.service")
        == f"{src}/gray/../phastform/systemd/flatshare-engine.service"
    )
    assert (
        os.readlink(units / "flatshare-gateway.service")
        == f"{src}/cyan/systemd/flatshare-gateway.service"
    )
    assert (
        os.readlink(units / "flatshare-engine.service.d" / "gray.conf")
        == f"{src}/gray/systemd/flatshare-engine.conf"
    )
    done = subprocess.run(
        ["bash", str(tree / "deploy" / "phastform" / "variant.sh"), "measure", "on"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert (
        os.readlink(units / "flatshare-engine.service.d" / "measure.conf")
        == f"{src}/phastform/measure/flatshare-engine.conf"
    )


def test_any_failure_after_the_switch_switches_back(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not only a failed check: an error of the tool's own after the switch, or an interrupt, goes back too.
    host.prepare(A, "v1.0")
    host.switch(A)
    host.start("box-a.service")
    host.prepare(B, "v1.1", **{"deploy/testhost/a.conf": "[Service]\nNice=5\n"})
    health = release.health
    calls = []

    def broken(*args, **kwargs):
        calls.append(release.Layout(host.root).target(host.layout.current))
        if len(calls) == 1:
            raise KeyError("a bug")
        return health(*args, **kwargs)

    monkeypatch.setattr(release, "health", broken)
    assert host.switch(B) == 1
    assert (host.current(), calls) == (A, [B, A])


UBUNTU_REFUSES = "unshare: write failed /proc/self/uid_map: Operation not permitted"
"""unshare's error where the kernel refuses unprivileged user namespaces (Ubuntu 24.04, Gray's, 2026-10-27)."""


def namespace_tools(bin_dir: Path, verdict: str) -> None:
    """unshare and ip stand-ins for deploy/bin/private-net-check: unshare runs its command (``allowed``) or fails as
    Gray's kernel makes it fail (``refused``); ``unknown`` leaves the tree without the check instead."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    run = (
        f"echo '{UBUNTU_REFUSES}' >&2\nexit 1\n"
        if verdict == "refused"
        else 'while [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n'
    )
    for name, text in (("unshare", f"#!/bin/sh\n{run}"), ("ip", "#!/bin/sh\nexit 0\n")):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)


@pytest.mark.parametrize("verdict", ["refused", "allowed", "unknown"])
def test_grays_install_runs_the_gpu_servers_in_the_hosts_namespace_where_the_kernel_refuses_their_own(
    tmp_path: Path, verdict: str
) -> None:
    # Gray's kernel refuses unprivileged user namespaces, where in-private-net fails a unit that asks for one: its
    # install links the host-network drop-in into each of the three GPU servers, through ~/flatshare/src, replacing the
    # hand-made one the engine had (2026-10-27); where they are allowed it removes it, and where the check cannot run
    # it changes nothing.
    home = tmp_path / "home"
    tree = home / "flatshare" / "releases" / A
    copied = ["gray/install.sh", "gray/private-net-host.conf", "phastform/systemd/flatshare.target"]
    copied += [
        f"gray/systemd/{unit}.conf"
        for unit in ("flatshare-engine", "flatshare-stt", "flatshare-tts")
    ]
    if verdict != "unknown":
        copied.append("bin/private-net-check")
    for rel in copied:
        (tree / "deploy" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEPLOY / rel, tree / "deploy" / rel)
    (home / "flatshare" / "src").symlink_to(f"releases/{A}")
    bin_dir = tmp_path / "bin"
    namespace_tools(bin_dir, verdict)
    for tool in ("systemctl", "loginctl"):
        (bin_dir / tool).write_text("#!/bin/sh\nexit 0\n")
        (bin_dir / tool).chmod(0o755)
    units = home / ".config" / "systemd" / "user"
    hand_made = units / "flatshare-engine.service.d" / "private-net-host.conf"
    hand_made.parent.mkdir(parents=True)
    hand_made.write_text("[Service]\nEnvironment=FLATSHARE_PRIVATE_NET=host\n")
    done = subprocess.run(
        ["bash", str(tree / "deploy" / "gray" / "install.sh")],
        env=os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    drop_ins = {
        unit: units / f"{unit}.service.d" / "private-net-host.conf"
        for unit in ("flatshare-engine", "flatshare-stt", "flatshare-tts")
    }
    src = home / "flatshare" / "src" / "deploy"
    if verdict == "refused":
        for path in drop_ins.values():
            assert os.readlink(path) == f"{src}/gray/private-net-host.conf"
            assert "Environment=FLATSHARE_PRIVATE_NET=host" in path.read_text().splitlines()
        assert UBUNTU_REFUSES in done.stdout and "host's network namespace" in done.stdout
    elif verdict == "allowed":
        assert not any(path.exists() or path.is_symlink() for path in drop_ins.values())
        assert "a network namespace of its own" in done.stdout
    else:
        assert (
            hand_made.read_text().endswith("FLATSHARE_PRIVATE_NET=host\n")
            and not hand_made.is_symlink()
        )
        assert not drop_ins["flatshare-stt"].exists() and not drop_ins["flatshare-tts"].exists()
        assert "left as they were" in done.stderr


class Environments(release.Systemd):
    """A user manager whose units have the environments given, and nothing else."""

    def __init__(self, envs: dict[str, dict[str, str]] | None = None) -> None:
        self.envs = envs or {}

    def environment(self, unit: str) -> dict[str, str]:
        return self.envs.get(unit, {})


def isolation_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str) -> Path:
    """A release tree with the real private-net-check and Gray's host-network drop-in, on a host whose kernel
    ``refused`` or ``allowed`` unprivileged user namespaces."""
    tree = tmp_path / "tree"
    for rel in ("bin/private-net-check", "gray/private-net-host.conf"):
        (tree / "deploy" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEPLOY / rel, tree / "deploy" / rel)
    namespace_tools(tmp_path / "bin", verdict)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    return tree


GPU = ("flatshare-engine.service", "flatshare-stt.service", "flatshare-tts.service")


def test_a_gpu_nodes_manifest_records_where_each_server_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Gray, whose kernel refuses user namespaces: its install links the host-network drop-in at the switch, so the
    # record says the host's namespace for all three, whatever the drop-ins said before (the engine's alone, by hand).
    tree = isolation_tree(tmp_path, monkeypatch, "refused")
    before = Environments({"flatshare-engine.service": {"FLATSHARE_PRIVATE_NET": "host"}})
    gray = release.isolation(release.HOSTS["gray"], before, tree)
    assert gray["user namespaces"].startswith(f"refused ({UBUNTU_REFUSES}")
    for unit in GPU:
        assert gray[unit].startswith("the host's network namespace (FLATSHARE_PRIVATE_NET=host)"), (
            unit
        )
    # Phastform's directory has no such drop-in: on a kernel that refused, its servers could not start, and say so.
    phastform = release.isolation(release.HOSTS["phastform"], Environments(), tree)
    assert all(phastform[unit].endswith("cannot start") for unit in GPU)
    assert release.isolation(release.HOSTS["cyan"], Environments(), tree) == {}


def test_where_namespaces_are_allowed_each_gpu_server_has_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = isolation_tree(tmp_path, monkeypatch, "allowed")
    for node in ("gray", "phastform"):
        record = release.isolation(release.HOSTS[node], Environments(), tree)
        assert record == dict.fromkeys(GPU, "a network namespace of its own") | {
            "user namespaces": "allowed"
        }


def test_a_release_whose_gpu_server_could_not_start_is_not_switched_to(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A GPU node whose kernel refuses user namespaces, without the host-network drop-in: verification fails a release
    # for a server that runs or would restart, since in-private-net would fail its start, and reports a stopped one.
    namespace_tools(host.tmp / "bin", "refused")
    gpu = release.HostSpec(
        "testhost",
        node="testhost",
        envs=SPEC.envs,
        units=SPEC.units,
        artifacts=SPEC.artifacts,
        gpu=True,
    )
    check = (DEPLOY / "bin" / "private-net-check").read_text()
    host.prepare(A, "v1.0", **{"deploy/bin/private-net-check": check})
    report = release.verify(host.layout, gpu, A, release.Systemd(), host.units)
    assert not report.failed, report.failed
    rows = {what: (ok, detail) for what, ok, detail in report.checks}
    assert rows["box-a.service network namespace"] == (
        True,
        "a network namespace of its own, which the kernel refuses here: the server cannot start",
    )
    host.start("box-a.service")
    report = release.verify(host.layout, gpu, A, release.Systemd(), host.units)
    assert report.failed == ["box-a.service network namespace"]
