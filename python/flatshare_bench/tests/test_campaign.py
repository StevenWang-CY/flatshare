"""benchmarks/e2/campaign.sh against stand-ins for the hosts: ssh runs each command in a directory per
host (fakes/campaign/ssh), rsync copies between them, and cyan's flatshare-bench and the GPU node's
variant.sh log what they are asked to do."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CAMPAIGN = REPO / "benchmarks" / "e2" / "campaign.sh"
FAKES = Path(__file__).parent / "fakes" / "campaign"
PHASTFORM = "phastform.example.org"  # a host by its full name: run.json names it phastform
DEPLOYED = "v0.3-30-g6851e13"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="campaign.sh needs bash")

VOICE = 'arm = 1\nlabel = "voice alone"\n'


@dataclass
class Hosts:
    root: Path
    """$FAKE_HOSTS: a home directory per host, by the name campaign.sh's ssh gives."""
    campaigns: Path
    """The campaign files, with their arm files under arms/."""
    results: Path
    """$FLATSHARE_E2_RESULTS."""

    def home(self, host: str) -> Path:
        return self.root / host

    def gpu(self, host: str, deployed: str | None = DEPLOYED) -> None:
        """A GPU node: its variant.sh, and the revision deploy.sh wrote unless ``deployed`` is None."""
        src = self.home(host) / "flatshare" / "src"
        (src / "deploy" / "phastform").mkdir(parents=True)
        shutil.copy(FAKES / "variant.sh", src / "deploy" / "phastform" / "variant.sh")
        if deployed is not None:
            (src / "DEPLOYED").write_text(deployed + "\n")

    def campaign(self, name: str, lines: list[str], arms: dict[str, str]) -> Path:
        for arm, text in arms.items():
            (self.campaigns / "arms" / f"{arm}.toml").write_text(text)
        path = self.campaigns / f"{name}.campaign"
        path.write_text("".join(f"{line}\n" for line in lines))
        return path

    def run(
        self,
        campaign: Path,
        *options: str,
        gpus: str = PHASTFORM,
        script: Path = CAMPAIGN,
        **extra: str,
    ) -> subprocess.CompletedProcess:
        env = (
            os.environ
            | extra
            | {
                "PATH": f"{FAKES}{os.pathsep}{os.environ['PATH']}",
                "FAKE_HOSTS": str(self.root),
                "FAKE_FLATSHARE_BENCH": str(Path(sys.executable).with_name("flatshare-bench")),
                "FLATSHARE_SSH": str(FAKES / "ssh"),
                "FLATSHARE_CYAN": "cyan",
                "FLATSHARE_GPU": gpus,
                "FLATSHARE_E2_RESULTS": str(self.results),
            }
        )
        return subprocess.run(
            ["bash", str(script), *options, str(campaign)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def calls(self, host: str, tool: str) -> list[str]:
        """What ``tool`` (flatshare-bench on cyan, variant.sh on a GPU node) was asked to do, call by call."""
        log = self.home(host) / f"{tool}.log"
        return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def hosts(tmp_path: Path) -> Hosts:
    hosts = Hosts(tmp_path / "hosts", tmp_path / "campaigns", tmp_path / "results")
    bench = hosts.home("cyan") / "flatshare" / "venvs" / "bench" / "bin"
    bench.mkdir(parents=True)
    shutil.copy(FAKES / "flatshare-bench", bench / "flatshare-bench")
    (hosts.campaigns / "arms").mkdir(parents=True)
    return hosts


def runs(hosts: Hosts) -> list[str]:
    """The points cyan's flatshare-bench measured, each as ``e2 run`` was called."""
    return [
        call
        for call in hosts.calls("cyan", "flatshare-bench")
        if call.startswith("flatshare-bench e2 run ")
    ]


def test_each_point_runs_with_the_gpu_nodes_deployed_revision(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c1", ["arms/a1.toml", "arms/a2.toml"], {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign)
    assert done.returncode == 0, done.stdout + done.stderr
    runs_of = (
        "flatshare-bench e2 run flatshare/runs/e2/c1/arms/{0}.toml --out flatshare/runs/e2/c1/{0}"
    )
    assert runs(hosts) == [
        f"{runs_of.format(point)} --gpu-source phastform={DEPLOYED}" for point in ("a1", "a2")
    ]
    # Each point's engine, with the measurement drop-in's routes, which flatshare-bench e2 run reads.
    switch = ["variant.sh measure on", "variant.sh use frozen"]
    assert hosts.calls(PHASTFORM, "variant") == switch * 2
    assert f"a1: phastform has {DEPLOYED}" in (hosts.results / "c1" / "campaign.log").read_text()
    # The GPU's samples end in its name and UUID, so that a record says which card it ran on; a node that names
    # no GPU of its own (FLATSHARE_GPU_ID) is sampled whole, as before.
    [query] = {call.split()[1] for call in hosts.calls(PHASTFORM, "nvidia-smi")}
    assert query.startswith("--query-gpu=timestamp,clocks.sm,") and query.endswith(",name,uuid")
    samples = (hosts.home("cyan") / "flatshare/runs/e2/c1/a1/gpu.csv").read_text().splitlines()
    assert samples and all(
        line.endswith(", NVIDIA GeForce RTX 5060 Ti, GPU-5060") for line in samples
    )


def test_a_host_that_lends_one_of_its_gpus_has_it_sampled_alone(hosts: Hosts) -> None:
    # A single host's env.sh names its lent GPU (FLATSHARE_GPU_ID), which the sampler, run there, reads: only that GPU
    # is sampled, and the file says which it is.
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c9", ["arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign, FLATSHARE_GPU_ID="GPU-5060")
    assert done.returncode == 0, done.stdout + done.stderr
    [call] = set(hosts.calls(PHASTFORM, "nvidia-smi"))
    assert call.startswith("nvidia-smi -i GPU-5060 --query-gpu=timestamp,")
    samples = (hosts.home("cyan") / "flatshare/runs/e2/c9/a1/gpu.csv").read_text().splitlines()
    assert samples[0] == "# gpu GPU-5060" and samples[1].endswith(", GPU-5060")


def test_flatshare_profile_names_the_gpu_nodes_profile_for_the_engine_check(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c6", ["arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign, FLATSHARE_PROFILE="a100-40gb")
    assert done.returncode == 0, done.stdout + done.stderr
    [run] = runs(hosts)
    assert run.endswith(f" --gpu-source phastform={DEPLOYED} --profile a100-40gb")
    done = hosts.run(campaign, FLATSHARE_PROFILE="../a100")
    assert done.returncode == 2 and "FLATSHARE_PROFILE is a profile's name" in done.stderr


def test_a_gpu_node_without_its_revision_runs_the_point_with_none(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM, deployed=None)
    campaign = hosts.campaign("c2", ["arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign)
    assert done.returncode == 0, done.stdout + done.stderr
    [run] = runs(hosts)
    assert run.endswith(" --gpu-source phastform=")
    assert "a1: phastform has no DEPLOYED" in (hosts.results / "c2" / "campaign.log").read_text()


def test_a_campaign_runs_on_one_gpu_node(hosts: Hosts) -> None:
    # The points on several GPU nodes are on the two-node branch.
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c9", ["arms/a1.toml"], {"a1": VOICE})
    for gpus in (f"{PHASTFORM},gray.example.org", f"{PHASTFORM} gray.example.org"):
        done = hosts.run(campaign, gpus=gpus)
        assert done.returncode == 2, gpus
        assert f"FLATSHARE_GPU names the GPU node's host, one: '{gpus}'" in done.stderr
    assert runs(hosts) == []


def test_a_campaign_sets_its_points_agents_seed(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    lines = ["# a round", "agents-seed 13  # this round's sample", "arms/a1.toml", "arms/a2.toml"]
    campaign = hosts.campaign("c3", lines, {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [run.split(" --gpu-source")[0].split()[-2:] for run in runs(hosts)] == [
        ["--agents-seed", "13"]
    ] * 2
    assert (
        "c3: 2 points, arm files at ?, the agents' seed 13"
        in (hosts.results / "c3" / "campaign.log").read_text()
    )


def test_a_campaign_on_a_copied_tree_names_the_commit_in_its_deployed(
    hosts: Hosts, tmp_path: Path
) -> None:
    # deploy.sh and push.sh copy the tree without .git, and write its commit into DEPLOYED at its top.
    tree = tmp_path / "src"
    (tree / "benchmarks" / "e2").mkdir(parents=True)
    shutil.copy(CAMPAIGN, tree / "benchmarks" / "e2" / "campaign.sh")
    (tree / "DEPLOYED").write_text("472adce\n")
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c11", ["arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign, script=tree / "benchmarks" / "e2" / "campaign.sh")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        "c11: 1 point, arm files at 472adce" in (hosts.results / "c11" / "campaign.log").read_text()
    )


@pytest.mark.parametrize(
    ("lines", "error"),
    [
        (
            ["agents-seed 11", "agents-seed 12"],
            "'agents-seed 12': a campaign sets the agents' seed once",
        ),
        (["agents-seed eleven"], "'agents-seed eleven': a campaign sets the agents' seed once"),
        (["agents-seed -1"], "'agents-seed -1': a campaign sets the agents' seed once"),
    ],
)
def test_a_campaign_refuses_a_seed_it_cannot_read(
    hosts: Hosts, lines: list[str], error: str
) -> None:
    campaign = hosts.campaign("c4", [*lines, "arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign, "--dry-run")
    assert done.returncode == 1 and error in done.stdout
    assert runs(hosts) == []


def test_a_campaign_sets_its_points_kv_sampler(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    lines = ["kv-sample-s 0  # as W5's points ran", "arms/a1.toml", "arms/a2.toml"]
    campaign = hosts.campaign("c8", lines, {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [run.split(" --gpu-source")[0].split()[-2:] for run in runs(hosts)] == [
        ["--kv-sample-s", "0"]
    ] * 2
    assert (
        "c8: 2 points, arm files at ?, no KV sampler"
        in (hosts.results / "c8" / "campaign.log").read_text()
    )


@pytest.mark.parametrize(
    ("lines", "error"),
    [
        (
            ["kv-sample-s 0", "kv-sample-s 1"],
            "'kv-sample-s 1': a campaign sets the KV sampler's interval once",
        ),
        (
            ["kv-sample-s fast"],
            "'kv-sample-s fast': a campaign sets the KV sampler's interval once",
        ),
    ],
)
def test_a_campaign_refuses_a_sampler_interval_it_cannot_read(
    hosts: Hosts, lines: list[str], error: str
) -> None:
    campaign = hosts.campaign("c9", [*lines, "arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign, "--dry-run")
    assert done.returncode == 1 and error in done.stdout
    assert runs(hosts) == []


def test_a_campaign_saves_its_calls_audio(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    lines = ["agents-seed 11", "save-audio  # for the demo", "arms/a1.toml", "arms/a2.toml"]
    campaign = hosts.campaign("c10", lines, {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [run.split(" --gpu-source")[0].split()[-3:] for run in runs(hosts)] == [
        ["--agents-seed", "11", "--save-audio"]
    ] * 2
    assert (
        "c10: 2 points, arm files at ?, the agents' seed 11, the calls' audio saved"
        in (hosts.results / "c10" / "campaign.log").read_text()
    )
    for line in (["save-audio", "save-audio"], ["save-audio twice"]):
        campaign = hosts.campaign("c11", [*line, "arms/a1.toml"], {"a1": VOICE})
        done = hosts.run(campaign, "--dry-run")
        assert done.returncode == 1 and "a campaign saves its calls' audio once" in done.stdout


def test_points_lists_a_campaigns_points_and_reaches_no_host(hosts: Hosts) -> None:
    # The A100 run's scripts read a campaign's points from here: each line that sets something for every point is no
    # point, and a file that is not a campaign is refused as a campaign refuses it.
    lines = [
        "# a round",
        "agents-seed 11",
        "kv-sample-s 0.5",
        "save-audio",
        "",
        "arms/a2.toml  # the stock arm",
        "arms/a1.toml",
    ]
    done = hosts.run(hosts.campaign("c14", lines, {"a1": VOICE, "a2": VOICE}), "--points")
    assert (done.returncode, done.stdout) == (0, "a2\na1\n"), done.stderr
    assert not (hosts.root / "ssh.log").exists() and not hosts.results.exists()  # no host, no log
    done = hosts.run(hosts.campaign("c15", ["save-audio twice", "arms/a1.toml"], {}), "--points")
    assert done.returncode == 1 and "a campaign saves its calls' audio once" in done.stderr
    assert done.stdout == ""


def test_campaign_sh_reads_the_demo_campaign(hosts: Hosts) -> None:
    done = hosts.run(REPO / "benchmarks" / "e2" / "demo.campaign", "--dry-run")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 points, arm files at" in done.stdout and "the calls' audio saved" in done.stdout


def test_without_a_seed_line_each_point_keeps_its_arm_files_seed(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    done = hosts.run(hosts.campaign("c5", ["arms/a1.toml"], {"a1": VOICE}))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--agents-seed" not in runs(hosts)[0]


@pytest.mark.parametrize("round", range(1, 6))
def test_campaign_sh_reads_each_w5_round(hosts: Hosts, round: int) -> None:
    done = hosts.run(REPO / "benchmarks" / "e2" / f"w5-r{round}.campaign", "--dry-run")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        "4 points, arm files at" in done.stdout and f"the agents' seed {10 + round}" in done.stdout
    )
    window = "--additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48"
    assert f"w5-disc-pr-ab48-v03-s24: pending, engine make-room-v03 {window}" in done.stdout
    assert f"w5-disc-pr-ab48-s24: pending, engine make-room {window}" in done.stdout
    assert "w5-a1-voice: pending, engine frozen" in done.stdout


def local_host(hosts: Hosts, node_dir: str = "flatshare/node") -> Path:
    """This host as cyan and the GPU node at once: flatshare-bench, the node's variant.sh under ``node_dir``,
    DEPLOYED."""
    home = hosts.home("localhost")
    (home / node_dir).mkdir(parents=True)
    shutil.copy(FAKES / "variant.sh", home / node_dir / "variant.sh")
    (home / "flatshare" / "src").mkdir(parents=True, exist_ok=True)
    (home / "flatshare" / "src" / "DEPLOYED").write_text(DEPLOYED + "\n")
    bench = home / "flatshare" / "venvs" / "bench" / "bin"
    bench.mkdir(parents=True)
    shutil.copy(FAKES / "flatshare-bench", bench / "flatshare-bench")
    return home


def test_a_local_campaign_runs_every_step_on_this_host(hosts: Hosts) -> None:
    home = local_host(hosts)
    campaign = hosts.campaign("c7", ["arms/a1.toml", "arms/a2.toml"], {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign, "--local", HOME=str(home), FLATSHARE_NODE_DIR="flatshare/node")
    assert done.returncode == 0, done.stdout + done.stderr
    # The node's own variant.sh switched the engine, with the measurement drop-in on, flatshare-bench measured here, and
    # nothing went over ssh.
    assert (
        hosts.calls("localhost", "variant")
        == ["variant.sh measure on", "variant.sh use frozen"] * 2
    )
    runs_of = (
        "flatshare-bench e2 run flatshare/runs/e2/c7/arms/{0}.toml --out flatshare/runs/e2/c7/{0}"
    )
    assert [
        c
        for c in hosts.calls("localhost", "flatshare-bench")
        if c.startswith("flatshare-bench e2 run ")
    ] == [f"{runs_of.format(point)} --gpu-source localhost={DEPLOYED}" for point in ("a1", "a2")]
    assert not (hosts.root / "ssh.log").exists()
    for point in ("a1", "a2"):
        assert (hosts.results / "c7" / point / "run.json").exists()
        assert (hosts.results / "c7" / point / "gpu.csv").read_text()
    bad = hosts.run(campaign, "--local", HOME=str(home), FLATSHARE_NODE_DIR="/etc")
    assert (
        bad.returncode == 2
        and "FLATSHARE_NODE_DIR is a directory under the GPU node's home" in bad.stderr
    )


def test_until_starts_no_point_that_would_end_after_it(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c8", ["arms/a1.toml"], {"a1": VOICE})
    # A voice-alone point and its engine switch take 9 minutes by W5's durations.
    soon = (datetime.now(UTC) + timedelta(minutes=5)).strftime("%H:%M")
    done = hosts.run(campaign, "--until", soon)
    assert done.returncode == 0, done.stdout + done.stderr
    assert runs(hosts) == []
    assert (
        f"a1: 9 minutes would end after {soon} UTC"
        in (hosts.results / "c8" / "campaign.log").read_text()
    )
    later = (datetime.now(UTC) + timedelta(minutes=60)).strftime("%H:%M")
    done = hosts.run(campaign, "--until", later)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(runs(hosts)) == 1
    assert hosts.run(campaign, "--until", "25:00").returncode == 2


def test_until_budgets_a_point_by_this_hosts_own_durations_once_one_has_run(hosts: Hosts) -> None:
    # W5's durations are cyan's: a host with fewer cores for AIPerf builds its dataset slower. Each point and engine
    # switch measured here goes to the results' durations.tsv, by kind, and --until budgets by the longest of a kind.
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c11", ["arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign)
    assert done.returncode == 0, done.stdout + done.stderr
    measured = [line.split() for line in (hosts.results / "durations.tsv").read_text().splitlines()]
    assert [kind for kind, _ in measured] == ["switch", "voice"]
    assert all(seconds.isdigit() for _, seconds in measured)
    # A voice point here took 25 minutes: one 20 minutes before the deadline is not started, where W5's 7 minutes
    # and a switch's 2 would have let it.
    (hosts.results / "durations.tsv").write_text("switch 60\nvoice 1500\n")
    second = hosts.campaign("c12", ["arms/a1.toml"], {"a1": VOICE})
    deadline = int(datetime.now(UTC).timestamp()) + 20 * 60
    done = hosts.run(second, "--until-epoch", str(deadline))
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(runs(hosts)) == 1  # c11's alone
    assert "a1: 26 minutes would end after" in (hosts.results / "c12" / "campaign.log").read_text()


def test_until_epoch_is_one_deadline_that_has_no_tomorrow(hosts: Hosts) -> None:
    # The A100 run hands every campaign its one deadline as an epoch: one that has passed stops every point, where
    # --until HH:MM would take tomorrow's; one far off runs them all.
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c10", ["arms/a1.toml"], {"a1": VOICE})
    past = int(datetime.now(UTC).timestamp()) - 60
    done = hosts.run(campaign, "--until-epoch", str(past))
    assert done.returncode == 0, done.stdout + done.stderr
    assert runs(hosts) == []
    log = (hosts.results / "c10" / "campaign.log").read_text()
    hhmm = datetime.fromtimestamp(past, UTC).strftime("%H:%M")
    assert f"until {hhmm} UTC (epoch {past})" in log
    assert f"a1: 9 minutes would end after {hhmm} UTC" in log
    done = hosts.run(campaign, "--until-epoch", str(past + 7200))
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(runs(hosts)) == 1
    both = hosts.run(campaign, "--until", "12:00", "--until-epoch", str(past))
    assert both.returncode == 2 and "--until or --until-epoch, not both" in both.stderr
    assert hosts.run(campaign, "--until-epoch", "soon").returncode == 2


STOP = "systemctl --user stop flatshare.target flatshare-engine flatshare-stt flatshare-tts"
"""What --pause asks of the GPU node's user manager."""
LEFT = "cyan's units and the GPU node stay as they are: a point may still be running on cyan"


def test_a_failed_campaign_puts_cyan_back_and_with_pause_stops_the_node(hosts: Hosts) -> None:
    # A failure ended the campaign before its end put cyan's units back and paused the GPU node, so
    # a failed campaign left the node running and cyan on the failed arm's overlay.
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c13", ["arms/a1.toml", "arms/a2.toml"], {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign, "--pause", FAKE_E2_FAIL="a2")
    assert done.returncode == 1
    log = (hosts.results / "c13" / "campaign.log").read_text()
    assert "a2 failed (exit status 1)" in log
    assert hosts.calls("cyan", "flatshare-bench")[-1] == "flatshare-bench e2 reset"
    assert hosts.calls(PHASTFORM, "systemctl") == [STOP]
    assert "phastform is paused" in log


def test_a_lost_connection_to_cyan_is_said_so_and_touches_nothing(hosts: Hosts) -> None:
    # A probe of whether a point is complete that lost its connection read as an incomplete point,
    # and the lock's probe after it as a point still running on cyan.
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c14", ["arms/a1.toml", "arms/a2.toml"], {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign, "--pause", FAKE_SSH_DOWN=r"cyan:^grep .*/a2/run\.json")
    assert done.returncode == 1
    log = (hosts.results / "c14" / "campaign.log").read_text()
    assert "a2: lost the connection to cyan (ssh exit status 255)" in log
    assert "still running" not in log.replace(LEFT, "")
    assert len(runs(hosts)) == 1  # a1's
    assert hosts.calls(PHASTFORM, "systemctl") == [] and LEFT in log


def test_a_point_still_running_on_cyan_is_left_to_end(hosts: Hosts) -> None:
    # The lock is held when a's point comes: a point of an earlier campaign whose ssh dropped runs
    # on, and neither cyan's units nor the node may change under it.
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c15", ["arms/a1.toml", "arms/a2.toml"], {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign, "--pause", FAKE_FLOCK_HELD="2")
    assert done.returncode == 1
    log = (hosts.results / "c15" / "campaign.log").read_text()
    assert "a point is still running on cyan" in log and LEFT in log
    assert "flatshare-bench e2 reset" not in hosts.calls("cyan", "flatshare-bench")
    assert hosts.calls(PHASTFORM, "systemctl") == []


def test_a_point_whose_connection_dropped_is_left_to_end(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c16", ["arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign, "--pause", FAKE_SSH_DOWN="cyan:e2 run ")
    assert done.returncode == 1
    log = (hosts.results / "c16" / "campaign.log").read_text()
    assert "a1: lost the connection to cyan; the point may still be running there" in log
    assert hosts.calls(PHASTFORM, "systemctl") == [] and LEFT in log


def test_pull_copies_a_single_hosts_results_and_runs(hosts: Hosts, tmp_path: Path) -> None:
    home = hosts.home("a100")
    (home / "flatshare" / "results" / "e2" / "core").mkdir(parents=True)
    (home / "flatshare" / "results" / "e2" / "core" / "campaign.log").write_text("done\n")
    (home / "flatshare" / "runs" / "probes").mkdir(parents=True)
    (home / "flatshare" / "runs" / "probes" / "kv.log").write_text("kv: priority-0 TTFT 140 ms\n")
    (home / "flatshare" / "node").mkdir()
    (home / "flatshare" / "node" / "preflight.json").write_text("{}\n")
    (home / "flatshare" / "cache").mkdir()
    (home / "flatshare" / "cache" / "warm.json").write_text('{"configurations": {}}\n')
    env = os.environ | {
        "PATH": f"{FAKES}{os.pathsep}{os.environ['PATH']}",
        "FAKE_HOSTS": str(hosts.root),
        "FLATSHARE_SSH": str(FAKES / "ssh"),
    }
    out = tmp_path / "pulled"
    done = subprocess.run(
        ["bash", str(REPO / "deploy" / "a100" / "pull.sh"), "--once", "a100", str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert (out / "results" / "e2" / "core" / "campaign.log").read_text() == "done\n"
    assert (out / "runs" / "probes" / "kv.log").read_text().startswith("kv:")
    # The node's records and the warm starts' come along; a cache file the host lacks is no failure.
    assert (out / "node" / "preflight.json").read_text() == "{}\n"
    assert (out / "cache" / "warm.json").exists() and not (
        out / "cache" / "jit-check.json"
    ).exists()
    assert "pull: a100 up to date" in (out / "pull.log").read_text()
