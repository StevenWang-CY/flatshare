"""The make-room variant's load check (src/leaseholder_vllm/make-room/tail.py): a failed call turn counts at the
timeout, as E2's report censors a reply that never came, the percentiles are inclusive, as E2's report computes them,
and the turns that failed and the agents that stopped early are printed. Its agents keep the frozen box's prompts and
count there and scale with another engine's pool and context; whole replies are timed to their 16th and last token;
a run against a stand-in engine prints what the PR bodies' renderer reads. It needs aiohttp, which the test extra and
the engine environment have; without it, these tests skip."""

import asyncio
import importlib.util
import json
import random
import re
import tempfile
from pathlib import Path

import pytest

aiohttp = pytest.importorskip("aiohttp")
web = pytest.importorskip("aiohttp.web")

TAIL = Path(__file__).resolve().parents[1] / "src/leaseholder_vllm/make-room/tail.py"


@pytest.fixture(scope="module")
def tail():
    spec = importlib.util.spec_from_file_location("lh_tail", TAIL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def box(tail):
    """The frozen box's engine: Qwen3-4B's 32,768-token context and a pool of 2,275 blocks of 16 tokens."""
    return tail.Engine("http://engine", "Qwen/Qwen3-4B-Instruct-2507-FP8", 32768, 36_400)


class Response:
    """An aiohttp response as tail.py reads it: a status, and a stream of server-sent lines."""

    def __init__(
        self, status: int, lines: list[bytes] | None = None, error: Exception | None = None
    ):
        self.status, self.lines, self.error = status, lines or [], error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def read(self) -> bytes:
        return b""

    @property
    def content(self):
        async def lines():
            for line in self.lines:
                yield line

        return lines()


class Session:
    def __init__(self, *responses: Response):
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.bodies.append(json)
        return self.responses.pop(0)


def streamed(*deltas: str) -> list[bytes]:
    chunks = [{"choices": [{"delta": {"content": delta}}]} for delta in deltas]
    return [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks] + [b"data: [DONE]\n"]


def streamed_with_usage(tokens: int, done: bool = True) -> list[bytes]:
    """A chat stream as vLLM sends it with continuous usage: the role, one chunk a token, the finish and the usage."""
    chunks = [
        {
            "choices": [{"delta": {"role": "assistant", "content": ""}}],
            "usage": {"completion_tokens": 0},
        }
    ]
    chunks += [
        {"choices": [{"delta": {"content": f" t{k}"}}], "usage": {"completion_tokens": k}}
        for k in range(1, tokens + 1)
    ]
    if done:
        chunks += [
            {
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {"completion_tokens": tokens},
            },
            {"choices": [], "usage": {"completion_tokens": tokens}},
        ]
    lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
    return lines + ([b"data: [DONE]\n"] if done else [])


def test_a_call_turn_fails_on_an_error_status_an_exception_or_the_timeout(tail):
    body = {"messages": [{"role": "user", "content": "hello"}]}

    def turn(response):
        return asyncio.run(tail.turn(Session(response), "http://engine", body))

    times, reply = turn(Response(200, streamed("Sure", ", done.")))
    assert times[0] is not None and reply == "Sure, done."
    assert turn(Response(500)) == ([None], "")
    assert turn(Response(200, error=aiohttp.ClientConnectionError())) == ([None], "")
    assert turn(Response(200, error=TimeoutError())) == ([None], "")
    assert turn(Response(200, [b"data: [DONE]\n"])) == ([None], "")  # no token at all


def test_the_calls_reach_an_engine_on_its_unix_socket(tail):
    seen = []

    async def chat(request):
        seen.append(request.path)
        response = web.StreamResponse()
        await response.prepare(request)
        for line in streamed("Sure", ", done."):
            await response.write(line)
        return response

    async def main():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        # A short path: a socket's must fit sun_path, which pytest's tmp_path may not.
        path = f"{tempfile.mkdtemp(prefix='lh-tail-')}/engine.sock"
        await web.UnixSite(runner, path).start()
        try:
            session, url = tail.connect(f"unix:{path}")
            async with session:
                return await tail.turn(
                    session, url, {"messages": [{"role": "user", "content": "hi"}]}
                )
        finally:
            await runner.cleanup()

    times, reply = asyncio.run(main())
    assert times[0] is not None and reply == "Sure, done."
    assert seen == ["/v1/chat/completions"]


def test_an_agent_stops_at_its_first_failed_request(tail, box):
    for response in (Response(503), Response(200, error=aiohttp.ClientConnectionError())):
        session = Session(response)
        ran = asyncio.run(tail.agent(session, box, 100, stop=float("inf"), fresh=True))
        assert ran is False and not session.responses


def test_the_summary_counts_failed_turns_at_the_timeout_and_stopped_agents(tail):
    ttfts = [100.0 + i for i in range(18)] + [None, None]
    assert tail.summary(ttfts, [True] * 11 + [False]) == (
        "calls: 20 turns, 2 failed (counted at 30,000 ms), TTFT p50 110 p90 3105 p95 ≥30000 max ≥30000 ms; "
        "agents: 1 of 12 stopped early"
    )
    assert tail.summary([100.0 + i for i in range(20)], [True]) == (
        "calls: 20 turns, 0 failed (counted at 30,000 ms), TTFT p50 110 p90 117 p95 118 max 119 ms; "
        "agents: 0 of 1 stopped early"
    )


def prompts(tail, monkeypatch, engine, seed: int, fresh: bool, requests: int) -> list[int]:
    """The prompt lengths an agent sends, its first ``requests`` requests answered and the next one failed."""
    monkeypatch.setattr(tail, "AGENT_THINK_S", (0, 0))
    session = Session(*[Response(200)] * requests, Response(503))
    assert asyncio.run(tail.agent(session, engine, seed, stop=float("inf"), fresh=fresh)) is False
    assert {body["model"] for body in session.bodies} == {engine.model}
    return [len(body["prompt"]) for body in session.bodies[:requests]]


def box_prompts(seed: int, fresh: bool, requests: int, scale: float = 1.0) -> list[int]:
    """The prompt lengths of tail.py's agent before the engine sized it, drawn as it drew them, each bound of its laws
    times ``scale``."""
    rng = random.Random(seed)

    def tokens(n: int) -> list[int]:
        return [rng.randrange(1000, 100_000) for _ in range(n)]

    def size(low: int, high: int) -> int:
        return rng.randrange(round(low * scale), round(high * scale))

    context, lengths = tokens(size(6000, 14000)), []
    for _ in range(requests):
        lengths.append(len(context))
        if fresh:
            context = tokens(size(6000, 30_000))
        else:
            context = context + tokens(size(1000, 4000))
            if len(context) > round(30_000 * scale):
                context = tokens(size(6000, 14000))
        rng.uniform(0, 3)
    return lengths


@pytest.mark.parametrize("fresh", [True, False])
def test_the_frozen_box_keeps_its_agents_and_their_prompts(tail, box, monkeypatch, fresh):
    assert (box.agents, box.prompt_scale) == (12, 1.0)
    for seed in (100, 111):
        assert prompts(tail, monkeypatch, box, seed, fresh, 30) == box_prompts(seed, fresh, 30)


@pytest.mark.parametrize("fresh", [True, False])
def test_a_short_context_shrinks_the_prompts_and_adds_agents(tail, monkeypatch, fresh):
    # A 3 GiB pool of Qwen3-4B's KV (1,365 blocks), its context cut to 16,384 tokens to fit it.
    engine = tail.Engine("http://engine", "m", 16384, 1365 * 16)
    assert engine.prompt_scale == pytest.approx(16336 / 30000)
    assert engine.agents == 13  # 12 x 0.6 of the box's pool, over the prompts' scale of 0.54
    lengths = prompts(tail, monkeypatch, engine, 100, fresh, 200)
    assert max(lengths) <= 16384 - tail.AGENT_TOKENS and min(lengths) >= 3267
    assert lengths == box_prompts(100, fresh, 200, 16336 / 30000)


def test_a_larger_pool_adds_agents(tail):
    # Qwen3-14B's pool on the A100 40GB (117,344 tokens, the forecast's), at a 32,768-token context.
    assert tail.Engine("http://engine", "m", 32768, 117_344).agents == 39


def test_a_whole_reply_is_timed_to_its_16th_and_last_token(tail):
    def turn(lines, marks):
        return asyncio.run(tail.turn(Session(Response(200, lines)), "http://engine", {}, marks))

    times, reply = turn(streamed_with_usage(40), (16, 40))
    assert None not in times and times == sorted(times)
    assert reply == "".join(f" t{k}" for k in range(1, 41))
    # A reply cut after 20 tokens reached its 16th but not its 40th.
    times, _ = turn(streamed_with_usage(20, done=False), (16, 40))
    assert times[0] is not None and times[1] is not None and times[2] is None
    # Without usage, each chunk with content counts one token.
    times, _ = turn(streamed(*[f" w{k}" for k in range(20)]), (16,))
    assert None not in times


def test_the_lines_of_a_run_with_whole_replies(tail):
    assert tail.timed((), [123.4]) == "ttft    123 ms"
    assert tail.timed((), [None]) == "failed"
    assert tail.timed((16, 40), [100.0, 400.0, 900.0]) == (
        "ttft    100 ms, 16 tokens    400 ms, 40 tokens    900 ms"
    )
    assert (
        tail.timed((16, 40), [100.0, 400.0, None])
        == "ttft    100 ms, 16 tokens    400 ms, 40 tokens failed"
    )
    turns = [[100.0, 300.0 + i, 900.0 + i] for i in range(19)] + [[100.0, 320.0, None]]
    assert tail.replies((16, 40), turns) == (
        "replies: 20 turns, 16 tokens p50 310 p90 317 p95 318 max 320 ms, "
        "40 tokens p50 910 p90 917 p95 2372 max ≥30000 ms"
    )
    assert tail.replies((16, 40), turns[:1]) == "replies: 1 turns"
    assert tail.throughput([10_000] * 30 + [20_000] * 18, 240) == (
        "agents: 48 requests in 240 s, 0.20 a second, with 2,750 prompt and 10 output tokens a second"
    )


def run(tail, monkeypatch, engine, *args) -> None:
    """tail.main against a stand-in engine, with the agents' thought and the calls' gaps shortened."""
    monkeypatch.setattr(tail, "AGENT_THINK_S", (0, 0.01))
    monkeypatch.setattr(tail, "CALL_START_S", (0, 0.05))
    monkeypatch.setattr(tail, "CALL_GAP_S", (0.05, 0.1))

    async def main():
        async with engine as url:
            await tail.main(url, *args)

    asyncio.run(main())


# What the PR bodies' renderer reads of a run: the turn lines, and the summary as the last line.
TURN = re.compile(r" call \d (?:ttft +(\d+) ms|failed)$")
SUMMARY = re.compile(r"^calls: (\d+) turns, (\d+) failed .*; agents: (\d+) of (\d+) stopped early$")


def test_a_run_prints_what_the_renderer_reads(tail, monkeypatch, capsys):
    from fake_engine import FakeEngine

    engine = FakeEngine()
    run(tail, monkeypatch, engine, 1.0, None, True)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "tail: 12 agents with fresh prompts of 6,000-30,000 tokens beside 2 calls with 16-token replies, for 1 s;"
        " Qwen/Qwen3-4B-Instruct-2507-FP8, a context of 32,768 tokens, a pool of 36,400"
    )
    turns = [match for line in lines if (match := TURN.search(line))]
    [summary] = [match for line in lines if (match := SUMMARY.match(line))]
    assert lines[-1] == summary.string and re.search(r"TTFT p50 \d+ p90 \d+ p95 \d+", lines[-1])
    assert int(summary[1]) == len(turns) >= 4 and summary.groups()[1:] == ("0", "0", "12")
    assert re.match(
        r"^agents: \d+ requests in 1 s, [\d.]+ a second, with [\d,]+ prompt and [\d,]+ output",
        lines[-2],
    )
    calls = [body for body in engine.bodies if "messages" in body]
    assert {body["max_tokens"] for body in calls} == {16} and not any(
        "ignore_eos" in body for body in calls
    )


def test_whole_replies_on_a_short_context(tail, monkeypatch, capsys):
    from fake_engine import FakeEngine

    engine = FakeEngine(num_gpu_blocks=1365, max_model_len=16384)
    run(tail, monkeypatch, engine, 1.0, None, False, 40)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(
        "tail: 13 agents with growing prompts of 3,267-16,336 tokens beside 2 calls with 40-token replies"
    )
    turns = [line for line in lines if " call " in line]
    assert turns and all(
        re.search(r"ttft +\d+ ms, 16 tokens +\d+ ms, 40 tokens +\d+ ms$", line) for line in turns
    )
    assert re.match(r"^replies: \d+ turns, 16 tokens p50 \d+ .* 40 tokens p50 \d+ ", lines[-3])
    assert lines[-1].endswith("; agents: 0 of 13 stopped early") and engine.rejected == 0
    calls = [body for body in engine.bodies if "messages" in body]
    assert all(body["max_tokens"] == 40 and body["ignore_eos"] for body in calls)
