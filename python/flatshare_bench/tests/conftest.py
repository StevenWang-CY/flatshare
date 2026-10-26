from __future__ import annotations

import os
import socket
import stat
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the caller's FLATSHARE_BENCH_* settings (the plugin's and the YAML template's) out of every test."""
    for name in [name for name in os.environ if name.startswith("FLATSHARE_BENCH_")]:
        monkeypatch.delenv(name)


@pytest.fixture
def socket_modes(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """The mode of each Unix socket's file as bind() creates it, by path, before its caller could change it; the
    process's umask is 022 meanwhile, a unit's default, under which a socket bound as it is would be mode 755."""
    modes: dict[str, int] = {}
    bind = socket.socket.bind

    def recorded(self: socket.socket, address: object) -> None:
        bind(self, address)
        if self.family == socket.AF_UNIX and isinstance(address, str):
            modes[address] = stat.S_IMODE(os.stat(address).st_mode)

    monkeypatch.setattr(socket.socket, "bind", recorded)
    mask = os.umask(0o022)
    try:
        yield modes
    finally:
        os.umask(mask)


@pytest.fixture
def fake_prompt_generator() -> MagicMock:
    """A tokenizer-free stand-in for AIPerf's PromptGenerator.

    Hash id h decodes to the corpus tokens at offset 64 * h, and a token t decodes to the word
    " w<t>", so whitespace splitting recovers the exact token sequence of a synthesized prompt.
    """
    pg = MagicMock()
    pg._tokenized_corpus = list(range(10_000, 14_096))
    pg._corpus_size = len(pg._tokenized_corpus)
    pg._cache = {}
    pg._bpe_stable_terminator_tokens = []
    state = {"hash_id": 0}
    pg._hash_id_corpus_rng.reseed_for_hash_id.side_effect = lambda h: state.update(hash_id=h)
    pg._hash_id_corpus_rng.randrange.side_effect = lambda n: (state["hash_id"] * 64) % n
    pg.tokenizer.decode.side_effect = lambda tokens: "".join(f" w{t}" for t in tokens)
    return pg
