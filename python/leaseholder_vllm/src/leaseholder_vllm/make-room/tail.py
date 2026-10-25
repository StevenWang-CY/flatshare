"""tail.py ENGINE SECONDS [AGENTS] [fresh|grow] [--reply TOKENS]: the make-room variant's load check, the calls' tail.

AGENTS closed-loop agent sessions at priority 2 beside two calls at priority 0, against one engine for SECONDS:
- agents: 48 output tokens and 0-3 s of thought a turn; with ``fresh``, a new prompt of 6-30k tokens each turn, as
  E2's discriminating replay computes about 95% of its prompt tokens; with ``grow`` (the default), the prompt grows
  by a tool result of 1-4k tokens each turn from 6-14k up to 30k, so that it reuses the CPU tier; an agent stops at
  its first failed request;
- calls: a conversation that grows by about 30 tokens a turn, a turn every 4-8 s, 16 output tokens; with --reply
  TOKENS, whole replies of exactly TOKENS tokens, each also timed to its 16th and its last token, so that the budget
  cap's window shows at the engine level.
The engine gives the model and its context (/v1/models) and its KV pool (its metrics). AGENTS, by default, keeps the
frozen box's KV pressure: 12 agents on its pool of 36,400 tokens, and in proportion on another pool; on a context too
short for a 30k prompt and its reply, the prompts shrink in proportion and the agents grow to match.
It prints the sizes, each call turn's time to first token (with --reply, to its 16th and last token too), then their
distribution, with each failed turn (an error, or no first token within TIMEOUT_MS) counted at TIMEOUT_MS, as E2's
report censors a reply that never came: a percentile it sets is marked ≥. It also prints the requests and tokens a
second the agents completed, how many turns failed and how many agents stopped early. On the GPU node, with the
engine's environment, which has aiohttp: ``~/leaseholder/venvs/engine/bin/python tail.py
unix:$XDG_RUNTIME_DIR/lh/engine.sock 240 12 fresh``. ENGINE is the engine's address: ``unix:`` and its socket, or
``http://host:port``.
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import time
from dataclasses import dataclass

import aiohttp

CALL = (
    "the order is late and the customer wants it moved to friday so check the address then confirm"
)
TIMEOUT_MS = 30_000.0  # E2's REPLY_TIMEOUT_MS
# A call's reply: 16 tokens, or with --reply a whole reply, timed to its 16th token as well.
FIRST_TOKENS = 16
AGENT_TOKENS = 48  # an agent's output tokens a turn
LONGEST_PROMPT = 30_000  # an agent's longest prompt
# The frozen box's agents, and its KV pool: 2,275 blocks of 16 tokens.
BOX_AGENTS, BOX_POOL_TOKENS = 12, 36_400
AGENT_THINK_S = (0, 3)  # an agent's thought after each turn
CALL_START_S, CALL_GAP_S = (0, 4), (4, 8)  # a call's first turn, then the time between its turns


@dataclass(frozen=True)
class Engine:
    """The engine under the check: its URL, its model and context (/v1/models), and its KV pool in tokens."""

    url: str
    model: str
    max_model_len: int
    pool_tokens: int

    @property
    def prompt_scale(self) -> float:
        """The agents' prompt sizes over the frozen box's: 1, unless the context cannot hold the longest and its reply."""
        return min(1.0, (self.max_model_len - AGENT_TOKENS) / LONGEST_PROMPT)

    @property
    def agents(self) -> int:
        """The agents that keep the frozen box's KV pressure: 12 on its pool, in proportion to the pool and inversely
        to the prompts' scale."""
        return max(1, round(BOX_AGENTS * self.pool_tokens / BOX_POOL_TOKENS / self.prompt_scale))


async def read_engine(session: aiohttp.ClientSession, url: str) -> Engine:
    """The engine's model and context, from /v1/models, and its pool, from its metrics."""
    async with session.get(f"{url}/v1/models") as response:
        card = (await response.json())["data"][0]
    async with session.get(f"{url}/metrics") as response:
        info = re.search(r"^vllm:cache_config_info\{([^}]*)\}", await response.text(), re.M)
    if info is None:
        raise RuntimeError(f"{url}/metrics has no vllm:cache_config_info")
    labels = dict(re.findall(r'(\w+)="([^"]*)"', info.group(1)))
    pool = int(labels["num_gpu_blocks"]) * int(labels["block_size"])
    return Engine(url, card["id"], int(card["max_model_len"]), pool)


def tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(1000, 100_000) for _ in range(n)]


async def agent(
    session: aiohttp.ClientSession,
    engine: Engine,
    seed: int,
    stop: float,
    fresh: bool,
    done: list[int] | None = None,
) -> bool:
    """Run an agent session until ``stop``; whether it got there, rather than stopping at a failed request. Each
    request it completes by ``stop`` adds its prompt's tokens to ``done``."""
    rng = random.Random(seed)
    scale = engine.prompt_scale

    def size(low: int, high: int) -> int:
        return rng.randrange(round(low * scale), round(high * scale))

    context = tokens(rng, size(6000, 14000))
    while time.time() < stop:
        body = {
            "model": engine.model,
            "prompt": context,
            "max_tokens": AGENT_TOKENS,
            "ignore_eos": True,
            "priority": 2,
        }
        try:
            async with session.post(f"{engine.url}/v1/completions", json=body) as response:
                await response.read()
                if response.status != 200:
                    return False
        except aiohttp.ClientError:
            return False
        if done is not None and time.time() <= stop:
            done.append(len(context))
        if fresh:  # E2's discriminating replay computes about 95% of its prompt tokens
            context = tokens(rng, size(6000, LONGEST_PROMPT))
        else:
            context = context + tokens(rng, size(1000, 4000))
            if len(context) > round(LONGEST_PROMPT * scale):
                context = tokens(rng, size(6000, 14000))
        await asyncio.sleep(rng.uniform(*AGENT_THINK_S))
    return True


async def turn(
    session: aiohttp.ClientSession, url: str, body: dict, marks: tuple[int, ...] = ()
) -> tuple[list[float | None], str]:
    """A call turn: its time in ms to its first token and to each token count in ``marks``, None for each it failed
    to reach (an error, or no such token within TIMEOUT_MS), and its reply. A chunk's usage counts the tokens so far;
    without usage, each chunk with content counts one."""
    started = time.time()
    times: list[float | None] = [None] * (1 + len(marks))
    reply: list[str] = []
    count = 0
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT_MS / 1000),
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if not line.startswith(b"data: {"):
                        continue
                    chunk = json.loads(line[6:])
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                    usage = chunk.get("usage")
                    count = usage.get("completion_tokens", count) if usage else count + bool(delta)
                    now = (time.time() - started) * 1000
                    if delta:
                        reply.append(delta)
                        if times[0] is None:
                            times[0] = now
                    for i, mark in enumerate(marks, 1):
                        if times[i] is None and count >= mark:
                            times[i] = now
    except (TimeoutError, aiohttp.ClientError):
        pass
    return times, "".join(reply)


def marks_for(reply: int | None) -> tuple[int, ...]:
    """The token counts a call turn is timed to after its first: none, or a whole reply's 16th and last."""
    return () if reply is None else (FIRST_TOKENS, reply)


def timed(marks: tuple[int, ...], times: list[float | None]) -> str:
    """A call turn's times as its line prints them."""
    if times[0] is None:
        return "failed"
    return ", ".join(
        [f"ttft {times[0]:6.0f} ms"]
        + [
            f"{mark} tokens " + ("failed" if t is None else f"{t:6.0f} ms")
            for mark, t in zip(marks, times[1:], strict=True)
        ]
    )


async def call(
    session: aiohttp.ClientSession,
    engine: Engine,
    seed: int,
    stop: float,
    turns: list[list[float | None]],
    reply: int | None = None,
) -> None:
    """Run a call until ``stop``, appending each turn's times in ms (to its first token, and with ``reply`` to its 16th
    and last), None where the turn failed to reach one."""
    rng = random.Random(seed)
    marks = marks_for(reply)
    messages = [
        {
            "role": "system",
            "content": f"You are a phone agent ({seed}). Answer in one short sentence.",
        }
    ]
    await asyncio.sleep(rng.uniform(*CALL_START_S))
    while time.time() < stop:
        messages.append({"role": "user", "content": " ".join(rng.choices(CALL.split(), k=20))})
        body = {
            "model": engine.model,
            "messages": messages,
            "max_tokens": FIRST_TOKENS,
            "priority": 0,
            "stream": True,
        }
        if reply is not None:
            body |= {
                "max_tokens": reply,
                "ignore_eos": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            }
        started = time.time()
        times, text = await turn(session, engine.url, body, marks)
        turns.append(times)
        at = f"{time.strftime('%H:%M:%S', time.gmtime(started))}.{int(started % 1 * 1000):03d}"
        print(f"{at} call {seed} {timed(marks, times)}", flush=True)
        messages.append({"role": "assistant", "content": text})
        await asyncio.sleep(rng.uniform(*CALL_GAP_S))


def ms(value: float) -> str:
    return ("≥" if value >= TIMEOUT_MS else "") + f"{value:.0f}"


def spread(samples: list[float]) -> str:
    """Sorted samples' p50, p90, p95 and max in ms, the percentiles inclusive, as E2's report computes them."""
    q = statistics.quantiles(samples, n=20, method="inclusive")
    return f"p50 {ms(statistics.median(samples))} p90 {ms(q[17])} p95 {ms(q[18])} max {ms(samples[-1])} ms"


def summary(ttfts: list[float | None], agents_ran: list[bool]) -> str:
    """The calls' first-token tail, a failed turn counted at TIMEOUT_MS (≥ marks a percentile one sets), and how
    many turns failed and agents stopped early. The percentiles are inclusive, as E2's report computes them."""
    samples = sorted(TIMEOUT_MS if ttft is None else ttft for ttft in ttfts)
    failed = sum(ttft is None for ttft in ttfts)
    stopped = sum(not ran for ran in agents_ran)
    agents = f"agents: {stopped} of {len(agents_ran)} stopped early"
    if len(samples) < 2:
        return f"calls: {len(samples)} turns, {failed} failed; {agents}"
    return (
        f"calls: {len(samples)} turns, {failed} failed (counted at {TIMEOUT_MS:,.0f} ms), TTFT {spread(samples)};"
        f" {agents}"
    )


def replies(marks: tuple[int, ...], turns: list[list[float | None]]) -> str:
    """The calls' times to each mark of their whole replies, a turn that failed to reach one counted at TIMEOUT_MS."""
    if len(turns) < 2:
        return f"replies: {len(turns)} turns"
    parts = [
        f"{mark} tokens {spread(sorted(TIMEOUT_MS if t[i] is None else t[i] for t in turns))}"
        for i, mark in enumerate(marks, 1)
    ]
    return f"replies: {len(turns)} turns, " + ", ".join(parts)


def throughput(done: list[int], seconds: float) -> str:
    """The agents' requests completed within the run, and their prompt and output tokens, a second."""
    return (
        f"agents: {len(done)} requests in {seconds:.0f} s, {len(done) / seconds:.2f} a second, with"
        f" {sum(done) / seconds:,.0f} prompt and {AGENT_TOKENS * len(done) / seconds:,.0f} output tokens a second"
    )


def connect(engine: str) -> tuple[aiohttp.ClientSession, str]:
    """A session to the engine at ``engine``, and the base of its requests' URLs: on the engine's socket, they reach
    it whatever their host."""
    socket = engine.removeprefix("unix:") if engine.startswith("unix:") else None
    connector = aiohttp.UnixConnector(path=socket) if socket else None
    session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None))
    return session, "http://localhost" if socket else engine


async def main(
    address: str, seconds: float, agents: int | None, fresh: bool, reply: int | None = None
) -> None:
    turns: list[list[float | None]] = []
    done: list[int] = []
    session, url = connect(address)
    async with session:
        engine = await read_engine(session, url)
        count = engine.agents if agents is None else agents
        scale = engine.prompt_scale
        prompts = f"{round(6000 * scale):,}-{round(LONGEST_PROMPT * scale):,}"
        print(
            f"tail: {count} agents with {'fresh' if fresh else 'growing'} prompts of {prompts} tokens beside 2 calls"
            f" with {reply or FIRST_TOKENS}-token replies, for {seconds:.0f} s; {engine.model}, a context of"
            f" {engine.max_model_len:,} tokens, a pool of {engine.pool_tokens:,}",
            flush=True,
        )
        stop = time.time() + seconds
        agents_ran = await asyncio.gather(
            *(agent(session, engine, 100 + i, stop, fresh, done) for i in range(count)),
            *(call(session, engine, i, stop, turns, reply) for i in (1, 2)),
        )
    marks = marks_for(reply)
    if marks:
        print(replies(marks, turns))
    print(throughput(done, seconds))
    print(summary([times[0] for times in turns], list(agents_ran[:count])))


def agent_count(value: str) -> int | None:
    if value == "pool":
        return None
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("AGENTS is a count of at least 1, or pool")
    return count


def reply_tokens(value: str) -> int:
    length = int(value)
    if length <= FIRST_TOKENS:
        raise argparse.ArgumentTypeError(f"a whole reply is longer than {FIRST_TOKENS} tokens")
    return length


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "engine", help="the engine's address: unix: and its socket, or http://host:port"
    )
    parser.add_argument("seconds", type=float)
    parser.add_argument(
        "agents", nargs="?", default="pool", type=agent_count, help="a count, or pool (the default)"
    )
    parser.add_argument("mode", nargs="?", default="grow", choices=("fresh", "grow"))
    parser.add_argument("--reply", type=reply_tokens, metavar="TOKENS", help="whole call replies")
    args = parser.parse_args()
    asyncio.run(main(args.engine, args.seconds, args.agents, args.mode == "fresh", args.reply))
