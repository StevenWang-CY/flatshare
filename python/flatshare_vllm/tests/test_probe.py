"""The make-room variant's probe (src/flatshare_vllm/make-room/probe.py): its requests sized from the engine's pool
and context, the frozen box's sizes unchanged, and its result line, against a stand-in engine on a loopback port and
on a Unix socket, the GPU node's, in the form that the committed probe outputs' check
(benchmarks/results/2026-10-23-make-room-probes/check.py) and the PR bodies' renderer parse. It needs aiohttp, which
the test extra and the engine environment have."""

import asyncio
import importlib.util
import re
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

ROOT = Path(__file__).resolve().parents[3]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return load(
        "flatshare_probe", ROOT / "python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py"
    )


@pytest.fixture(scope="module")
def check():
    return load(
        "flatshare_probe_check", ROOT / "benchmarks/results/2026-10-23-make-room-probes/check.py"
    )


# The result line in full, as the PR bodies' renderer reads it: case, TTFT, 16 tokens, the voice turn's prompt, the
# priority-2 requests and their prompt, the free blocks and the pool, the state after the turn.
LINE = re.compile(
    r"^(kv|prefill): priority-0 TTFT (\d+) ms, 16 tokens (\d+) ms, for (\d+) prompt tokens, sent with (\d+) "
    r"priority-2 requests of (\d+) tokens running and (\d+) of (\d+) blocks free; after it, (\d+) running and "
    r"(\d+) waiting$"
)


def test_the_frozen_box_keeps_its_sizes(probe):
    # 2,275 blocks of 16 tokens and a 32,768-token context: two of 17,999 tokens leave 24 blocks free; three of 10,000.
    assert probe.background("kv", 2275, 16, 32768) == (2, 17_999)
    assert probe.background("prefill", 2275, 16, 32768) == (3, 10_000)
    assert probe.FIRST_TOKENS == 16 and not hasattr(probe, "FIRST_SENTENCE")


@pytest.mark.parametrize(
    "num_gpu_blocks, max_model_len, kv, prefill",
    [
        # A 3 GiB pool of Qwen3-4B's KV, its context cut to fit.
        (1365, 16384, (2, 10_719), (3, 6_000)),
        # A 4.75 GiB pool.
        (2161, 32768, (2, 17_087), (3, 9_488)),
        # Qwen3-14B's 117,344 tokens on the A100 40GB (the forecast's).
        (7334, 32768, (5, 23_375), (4, 24_160)),
        # The box's pool behind a short context.
        (2275, 8192, (9, 3_999), (8, 3_744)),
    ],
)
def test_another_pool_or_context_sizes_the_requests_from_it(
    probe, num_gpu_blocks, max_model_len, kv, prefill
):
    assert probe.background("kv", num_gpu_blocks, 16, max_model_len) == kv
    assert probe.background("prefill", num_gpu_blocks, 16, max_model_len) == prefill
    count, prompt = kv
    # Each request fits its context with its decode, and once each has its first token, 24 blocks or fewer are free.
    assert prompt + probe.DECODE_TOKENS <= max_model_len
    free = num_gpu_blocks - 1 - count * ((prompt + 1 + 15) // 16)
    assert 0 <= free - probe.FREE_BLOCKS < count
    count, prompt = prefill
    assert prompt + probe.DECODE_TOKENS <= max_model_len
    # The prefills take about three times 625 blocks of every 2,275, whatever the count.
    share = count * prompt / 16 / (num_gpu_blocks * 3 * 625 / 2275)
    assert 0.99 < share <= 1.0


def test_a_context_without_room_for_the_decode_is_refused(probe):
    with pytest.raises(ValueError, match="no room"):
        probe.background("kv", 2275, 16, 4096)


def run(probe, engine, case: str, reply: int | None = None) -> None:
    async def main():
        async with engine as url:
            await probe.main(url, case, reply)

    asyncio.run(main())


@pytest.mark.parametrize("case", ["kv", "prefill"])
def test_the_result_line_still_parses(probe, check, capsys, case):
    from fake_engine import FakeEngine

    engine = FakeEngine()
    run(probe, engine, case)
    [line] = capsys.readouterr().out.splitlines()
    match = LINE.match(line)
    assert match, line
    background = {"kv": ("2", "17999"), "prefill": ("3", "10000")}[case]
    assert match.group(1) == case and (match.group(5), match.group(6)) == background
    assert match.group(8) == "2274"
    if case == "kv":
        assert int(match.group(7)) <= probe.FREE_BLOCKS
    # check.py's pattern, which reads the committed outputs, reads it too.
    assert check.RUN.match(line), line
    # The model's name came from the engine, and every request carried it.
    assert {body["model"] for body in engine.bodies} == {engine.model}
    voice = engine.bodies[-1]
    assert voice["priority"] == 0 and voice["max_tokens"] == 32 and "ignore_eos" not in voice


def test_a_whole_reply_is_timed_at_its_last_token(probe, capsys):
    from fake_engine import FakeEngine

    engine = FakeEngine()
    run(probe, engine, "prefill", reply=40)
    [line] = capsys.readouterr().out.splitlines()
    match = re.match(
        r"^prefill: priority-0 TTFT (\d+) ms, 16 tokens (\d+) ms, 40 tokens (\d+) ms, for \d+ ",
        line,
    )
    assert match, line
    ttft, t16, t40 = (int(group) for group in match.groups())
    assert ttft <= t16 <= t40
    voice = engine.bodies[-1]
    assert voice["max_tokens"] == 40 and voice["ignore_eos"] is True


def test_the_probe_reaches_an_engine_on_its_unix_socket(probe, capsys):
    from fake_engine import FakeEngine

    # A short path: a socket's must fit sun_path, which pytest's tmp_path may not.
    engine = FakeEngine(socket=f"{tempfile.mkdtemp(prefix='flatshare-probe-')}/engine.sock")
    run(probe, engine, "kv")
    [line] = capsys.readouterr().out.splitlines()
    assert LINE.match(line), line
    assert engine.bodies and {body["model"] for body in engine.bodies} == {engine.model}
    # By default it takes the GPU node's engine socket.
    runtime = probe.os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{probe.os.getuid()}"
    assert f"unix:{runtime}/flatshare/engine.sock" == probe.ENGINE


def test_a_large_pool_fills_with_more_requests(probe, capsys):
    from fake_engine import FakeEngine

    engine = FakeEngine(num_gpu_blocks=7334)
    run(probe, engine, "kv")
    [line] = capsys.readouterr().out.splitlines()
    match = LINE.match(line)
    assert match, line
    assert (match.group(5), match.group(6), match.group(8)) == ("5", "23375", "7333")
    assert int(match.group(7)) <= probe.FREE_BLOCKS
    assert sum(body["priority"] == 2 for body in engine.bodies) == 5
