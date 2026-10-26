"""The ``flatshare-voice`` command line refuses what the callers cannot run with, before anything starts."""

from __future__ import annotations

from pathlib import Path

import pytest

from flatshare_voice import cli
from flatshare_voice.bots import BotsConfig

BOTS = ["bots", "--corpus", "corpus", "--out", "run"]


def test_bots_takes_its_settings_from_the_command_line() -> None:
    args = cli.build_parser().parse_args(
        [*BOTS, "--callers", "6", "--turns", "10", "--barge-in", "0", "--warmup-turns", "1"]
    )
    assert (args.callers, args.turns, args.barge_in, args.warmup_turns) == (6, 10, 0.0, 1)
    assert (args.corpus, args.out) == (Path("corpus"), Path("run"))


@pytest.mark.parametrize(
    ("option", "value", "says"),
    [
        ("--callers", "0", "must be 1 or more, got 0"),
        ("--turns", "-3", "must be 1 or more, got -3"),
        ("--barge-in", "1.5", "must be a share from 0 to 1, got 1.5"),
        ("--warmup-turns", "-1", "must be 0 or more, got -1"),
        ("--callers", "two", "invalid _positive value: 'two'"),
    ],
)
def test_bots_refuses_settings_no_call_can_run_with(option, value, says, capsys) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli.build_parser().parse_args([*BOTS, option, value])
    assert exit_.value.code == 2
    assert f"argument {option}: {says}" in capsys.readouterr().err


def test_bots_names_the_livekit_credentials_it_lacks(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    with pytest.raises(SystemExit) as exit_:
        cli.main(BOTS)
    assert exit_.value.code == 2
    assert "bots needs LIVEKIT_API_SECRET in the environment" in capsys.readouterr().err
    monkeypatch.delenv("LIVEKIT_API_KEY")
    with pytest.raises(SystemExit):
        cli.main(BOTS)
    assert "LIVEKIT_API_KEY and LIVEKIT_API_SECRET" in capsys.readouterr().err


def test_a_bots_configuration_holds_the_same_rules() -> None:
    for wrong, says in [
        ({"callers": 0}, "callers and turns must be 1 or more"),
        ({"turns": 0}, "callers and turns must be 1 or more"),
        ({"barge_in": -0.1}, "barge_in is a share of the turns"),
        ({"warmup_turns": -1}, "warmup_turns cannot be negative"),
    ]:
        with pytest.raises(ValueError, match=says):
            BotsConfig("ws://fake", "key", "secret", Path("."), **wrong)
