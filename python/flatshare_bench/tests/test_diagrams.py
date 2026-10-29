"""The system diagrams (flatshare_bench.diagrams): each draws in both layouts with square corners, passes the style's
checks and says the same in each, and what it shows is what the files it depicts say. The request path's steps run in
the gateway's order with the refusals of its README's error table; the release's states and exits are the release
tool's and the runbook's; the triage's rows are alerts whose runbook URLs open them; the two cards' bounds are the
example gateway.toml's and their failover the drills' record's; and the architecture's sockets, listener and
namespaces are the units' and the gateway's. That every SVG under docs/figures is what its generator draws today,
test_figures.py checks with every other figure."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest

from flatshare_bench import diagrams

REPO = Path(__file__).resolve().parents[3]
SVG = "{http://www.w3.org/2000/svg}"
GATEWAY = REPO / "crates" / "flatshare-gateway"
DEPLOY = REPO / "deploy"
RUNBOOK = REPO / "ops" / "RUNBOOK.md"


def words(path: Path) -> set[str]:
    """The words of a figure's text, its labels' and not its title's or description's."""
    root = ET.parse(path).getroot()
    text = " ".join("".join(t.itertext()) for t in root.iter(f"{SVG}text"))
    return set(re.findall(r"[\w./:@-]+|[\w]+", text))


@pytest.mark.parametrize("diagram", diagrams.SYSTEM, ids=lambda d: d.__name__)
def test_each_diagram_draws_both_layouts_square_and_says_the_same_in_each(
    diagram: Callable[[Path], list[Path]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Draft mode would write a figure that fails its checks; nothing but a person laying one out sets it.
    monkeypatch.delenv("FLATSHARE_FIGURE_DRAFT", raising=False)
    wide, narrow = diagram(tmp_path)
    assert narrow.name == f"{wide.stem}-narrow.svg"
    for path, width in ((wide, 1120), (narrow, 420)):
        root = ET.parse(path).getroot()
        assert float(root.get("width")) == width, path.name
        title, description = root.find(f"{SVG}title").text, root.find(f"{SVG}desc").text
        # The alt text: a name, and one sentence that says what the figure shows.
        assert title and not title.endswith("."), title
        assert description.endswith(".") and not re.search(r"\. [A-Z]", description), description
        # Square corners: no box, group, panel or cell is rounded.
        assert " rx=" not in path.read_text(), path.name
    # The phone layout breaks its lines elsewhere and drops no label.
    assert words(wide) == words(narrow), words(wide) ^ words(narrow)


def error_table() -> set[tuple[str, str]]:
    """Every (status, code) the gateway README's error table gives."""
    readme = (GATEWAY / "README.md").read_text()
    section = readme.split("## Errors", 1)[1].split("\n## ", 1)[0]
    found = set()
    for row in re.findall(r"^\| ([\d, ]+) \| (.+?) \|", section, re.M):
        for status in row[0].split(", "):
            found |= {(status, code) for code in re.findall(r"`(\w+)`", row[1])}
    return found


def test_the_request_path_is_the_gateways_order_and_its_refusals_its_error_table() -> None:
    table = error_table()
    readme = (GATEWAY / "README.md").read_text()
    [types] = re.findall(r"Its `type` names the failure \(([^)]*)\)", readme)
    events = set(re.findall(r"`(\w+)`", types))
    assert events == {"upstream_unavailable", "upstream_timeout", "draining"}
    for step in diagrams.REQUEST_STEPS:
        for lead, code in step.answers:
            if lead:
                assert (lead, code) in table, (step.name, lead, code)
            else:  # once a stream has begun: its error event's type
                assert step.name == "Relay" and code in events, (step.name, code)
    # The steps run in the order the proxy takes a request: the key and the class, the bounds before the body, the
    # body read and rewritten (its fields sanitized, its reply capped) and the session resolved, the placement, and
    # the forward, whose relay and accounting follow.
    proxy = (GATEWAY / "src" / "proxy.rs").read_text()
    inference = proxy.split("pub(crate) async fn inference(", 1)[1].split(
        "\npub(crate) async fn", 1
    )[0]
    calls = [
        "authenticate(",
        "resolve_class(",
        "intake.admit(",
        "read_body(",
        "rewrite_body(",
        "resolve_session(",
        "nodes.request(",
        "forward(&app",
    ]
    at = [inference.index(call) for call in calls]
    assert at == sorted(at), dict(zip(calls, at, strict=True))
    assert [step.name for step in diagrams.REQUEST_STEPS] == [
        "Authenticate",
        "Bound",
        "Sanitize",
        "Place",
        "Forward",
        "Relay",
        "Account",
    ]


def test_the_release_is_the_release_tools_states_and_exits() -> None:
    tool = (DEPLOY / "bin" / "flatshare-release").read_text()
    assert (
        "Exit status: 0 on success; 1 when a check fails or a switch was switched back; 2 when it is used wrongly or "
        "the host\nis not ready (another operation holds the lock, an upload that differs from its hash, an old layout "
        "without --adopt);\n3 when switching back failed too."
    ) in tool
    # The verification's failure is a check's (exit 1), the drain's a host not ready (exit 2), and both switch nothing.
    assert "fails its checks ({', '.join(checks.failed)}): nothing switched\"" in tool
    assert 'raise NotReady(\n                f"the voice worker still has {calls} call(s)' in tool
    assert (
        'raise NotReady(\n            f"an E2 point (pid {pid}) measures on this host: nothing switched'
        in tool
    )
    exits = {step.name: step.answers for step in diagrams.RELEASE_STEPS if step.answers}
    assert exits == {
        "Prepare": (("exit 2", "nothing switched"),),
        "Verify": (("exit 1", "nothing switched"),),
        "Drain": (("exit 2", "nothing switched"),),
    }
    assert [lead for lead, _ in diagrams.SWITCH_BACK] == ["exit 1", "exit 3"]
    assert 'say(f"the host runs {old} again")\n        return 1' in tool
    assert 'say(f"switching back to {old} failed too: ' in tool
    # The runbook's steps, in its order.
    steps = re.findall(r"^\d\. \*\*([A-Z][a-z]+)", RUNBOOK.read_text(), re.M)[:6]
    assert steps == ["Prepare", "Verify", "Drain", "Switch", "Restart", "Check"]
    assert [step.name for step in diagrams.RELEASE_STEPS] == steps


def test_the_triages_rows_are_alerts_whose_runbook_urls_open_them() -> None:
    rules = (REPO / "ops" / "prometheus" / "rules.yml").read_text()
    anchors = set(re.findall(r'<a id="([^"]+)">', RUNBOOK.read_text()))
    named = [code for step in diagrams.TRIAGE_STEPS for _, code in step.answers]
    alerts = [name for name in named if name.startswith("Flatshare")]
    assert len(alerts) == len(named) - 1
    for alert in alerts:
        url = re.search(rf"- alert: {alert}\n(?:.*\n)*?\s+runbook_url: \S+#(\S+)", rules)
        assert url and url[1] == alert.lower() and url[1] in anchors, alert
    # The one row without an alert: agents' requests that find every node at its bound.
    assert "503 no_agent_slot" in named
    assert "agents' requests get 503 `no_agent_slot`" in RUNBOOK.read_text()


def commented_nodes() -> list[tuple[str, int, int]]:
    """The example gateway.toml's two-card [[nodes]], as (name, max_calls, max_agents)."""
    example = (DEPLOY / "cyan" / "gateway.example.toml").read_text()
    blocks = example.split("# [[nodes]]")[1:]
    found = []
    for block in blocks:
        name = re.search(r'^# name = "(\w+)"', block, re.M)[1]
        calls = int(re.search(r"^# max_calls = (\d+)", block, re.M)[1])
        agents = int(re.search(r"^# max_agents = (\d+)", block, re.M)[1])
        found.append((name, calls, agents))
    return found


def test_the_two_cards_are_the_example_nodes_and_the_failover_drills() -> None:
    assert [(n.lower(), c, a) for n, c, a in diagrams.CARDS] == commented_nodes()
    example = (DEPLOY / "cyan" / "gateway.example.toml").read_text()
    assert '# calls = "fill"' in example
    # The spare covers while it carries the calls: its agents' requests in flight held to none, until they end.
    assert "# failover_background_max_in_flight = 0" in example
    gateway = (REPO / "crates" / "flatshare-gateway" / "README.md").read_text()
    assert "are held to `failover_background_max_in_flight`, 0 by default" in gateway
    assert "until the moved calls end" in gateway
    assert diagrams.HELD == "held until the calls end"
    record = (
        REPO / "benchmarks" / "results" / "2026-10-28-node-failover" / "README.md"
    ).read_text()
    rows = [
        line.strip("|").split("|")
        for line in record.splitlines()
        if line.startswith(("| kill", "| stop"))
    ]
    assert {row[0].strip(): row[3].strip() for row in rows} == {"kill": "+0.7 s", "stop": "+1.9 s"}
    assert " ".join(diagrams.MOVED) == "next turn: 0.7 s after a kill, 1.9 s after a stop"


def test_the_architecture_is_the_units_sockets_listeners_and_namespaces() -> None:
    servers = ("engine", "stt", "tts")
    assert [socket for _, socket, _ in diagrams.SERVERS] == [f"{name}.sock" for name in servers]
    for name in servers:
        unit = (DEPLOY / "phastform" / "systemd" / f"flatshare-{name}.service").read_text()
        # Its socket in the user's flatshare directory, and the server in a network namespace of its own.
        assert f"%t/flatshare/{name}.sock" in unit, name
        assert "ExecStart=%h/flatshare/src/deploy/bin/in-private-net " in unit, name
    # Gray's servers run in the host's namespace, where its kernel refuses unprivileged user namespaces.
    assert (
        "Environment=FLATSHARE_PRIVATE_NET=host"
        in (DEPLOY / "gray" / "private-net-host.conf").read_text()
    )
    assert [node[0] for node in diagrams.NODES] == ["Phastform", "Gray"]
    # The one TCP listener, and the voice plane's socket, which its worker sends its LLM requests to; the worker's
    # speech goes to the node's sockets on cyan, which the node's tunnel serves.
    cyan = DEPLOY / "cyan"
    socket_unit = (cyan / "systemd" / "flatshare-gateway.socket").read_text()
    assert re.findall(r"^ListenStream=(.+)$", socket_unit, re.M) == [
        "127.0.0.1:18080",
        "%t/flatshare/gateway.sock",
    ]
    example = (cyan / "gateway.example.toml").read_text()
    assert re.findall(r'^tcp = "(.+)"$', example, re.M) == ["127.0.0.1:18080"]
    # Its requests need a key, a tenant's: the agents' tenant has its keys' digests.
    assert "Other users of cyan can reach loopback, so requests here need a key." in example
    assert re.search(r"^\[tenants\.agents\]\nkeys_sha256 = \[", example, re.M)
    assert diagrams.KEY == "TCP, a tenant's key"
    worker = (cyan / "systemd" / "flatshare-voice-worker.service").read_text()
    assert "Environment=FLATSHARE_LLM_UDS=%t/flatshare/gateway.sock" in worker
    assert "Environment=FLATSHARE_STT_UDS=%t/flatshare/phastform-stt.sock" in worker
    assert "Environment=FLATSHARE_TTS_UDS=%t/flatshare/phastform-tts.sock" in worker
    tunnel = (DEPLOY / "bin" / "flatshare-tunnel").read_text()
    assert 'SOCKETS = ("engine", "stt", "tts")' in tunnel
    # The voice plane in its namespace: LiveKit and the worker run in it.
    for unit in ("flatshare-livekit.service", "flatshare-voice-worker.service"):
        assert (
            "ExecStart=%h/flatshare/src/deploy/bin/in-voicenet "
            in (cyan / "systemd" / unit).read_text()
        )
