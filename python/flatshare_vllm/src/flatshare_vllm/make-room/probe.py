"""probe.py [ENGINE] [kv|prefill] [--reply TOKENS]: the make-room variant's two checks, a voice turn behind agent requests.

kv (the default): priority-2 requests fill the engine's KV pool, leaving 24 blocks free, and keep decoding; then a
priority-0 chat turn of about 750 tokens (47 blocks) arrives. Without make-room it waits until they have grown into the
free blocks and vLLM preempts one of them so that another can grow (a few seconds); with it, it preempts one at once.
The probe sends the voice turn once the engine reports them all running and the pool full.

prefill: three priority-2 requests of 625 blocks for every 2,275 of the pool prefill, in chunks beside each other,
with room in the pool for the voice turn, which arrives once all of them run. Its steps carry their chunks, each slower
at a longer context, unless the budget cap keeps lower-priority prefill out of the steps of its first tokens.

The sizes come from the engine: the pool from its metrics, the model and its context from /v1/models. Each priority-2
request leaves room in the context for its 4,096 decoded tokens, so where two (kv) or three (prefill) could not, more
requests share the same blocks. On the frozen box's pool of 2,275 blocks, kv sends two of 17,999 tokens and prefill
three of 10,000.

The probe prints the voice turn's time to its first token and to its 16th, and the engine's state before and after
it. With --reply TOKENS (more than 16), the voice turn's reply is exactly TOKENS tokens long and the time to its last
token follows the 16th's. The reserve (make_room.reserve_tokens) keeps the last priority-2 request out of a full pool,
so the KV case is run with it off. On the GPU node, with the engine's environment, which has aiohttp:
``~/flatshare/venvs/engine/bin/python probe.py unix:$XDG_RUNTIME_DIR/flatshare/engine.sock CASE`` on the frozen box,
then after ``deploy/phastform/variant.sh use make-room --additional-config.make_room.reserve_tokens 0``. ENGINE is the
engine's address: ``unix:`` and its socket (the default, the GPU node's), or ``http://host:port``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import aiohttp

# flatshare-stats (python/flatshare_stats), the repository's one definition of the statistics it publishes, from
# this source tree: the engine's environment, where the probe runs, does not install it.
_STATS = importlib.util.spec_from_file_location(
    "flatshare_stats",
    Path(__file__).resolve().parents[4] / "flatshare_stats/src/flatshare_stats/__init__.py",
)
stats = importlib.util.module_from_spec(_STATS)
_STATS.loader.exec_module(stats)

RUNTIME = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
ENGINE = f"unix:{RUNTIME}/flatshare/engine.sock"  # the GPU node's engine socket
FREE_BLOCKS = 24  # kv: left free by the priority-2 prompts; their decodes grow into them
# prefill: each priority-2 prompt takes 625 blocks for every 2,275 of the pool, 10,000 tokens on the frozen box.
PREFILL_BLOCKS, BOX_BLOCKS = 625, 2275
BACKGROUND = {"kv": 2, "prefill": 3}  # the fewest priority-2 requests of each case
# The priority-2 requests decode until the probe ends. With a short reply the first would finish while the second
# still prefills, and the voice turn would find half the pool free.
DECODE_TOKENS = 4096
FIRST_TOKENS = 16  # the voice turn is timed to its first token and to its 16th
CACHE_CONFIG = "vllm:cache_config_info"  # the pool's blocks and their size, as labels
CALL = (
    "the call is about a late order and the customer wants it moved to friday so check the address"
    " then confirm the time window and read back the total with tax before you say goodbye"
)


def background(
    case: str, num_gpu_blocks: int, block_size: int, max_model_len: int
) -> tuple[int, int]:
    """The priority-2 requests' count and prompt tokens, for a pool of ``num_gpu_blocks`` blocks as vLLM reports it
    (its null block included) and a context of ``max_model_len`` tokens. A prompt leaves room in the context for its
    decoded tokens; where the fewest requests could not, more of them share the same blocks."""
    longest = (max_model_len - DECODE_TOKENS) // block_size  # the most blocks a prompt may take
    if longest < 1:
        raise ValueError(
            f"a context of {max_model_len} tokens leaves no room for {DECODE_TOKENS} decoded tokens"
        )
    if case == "kv":
        fill = num_gpu_blocks - 1 - FREE_BLOCKS  # less vLLM's null block
        count = max(BACKGROUND["kv"], -(-fill // longest))
        # A prompt one token short of k blocks fills them with its first decoded token.
        return count, fill // count * block_size - 1
    blocks = BACKGROUND["prefill"] * (num_gpu_blocks * PREFILL_BLOCKS // BOX_BLOCKS)
    count = max(BACKGROUND["prefill"], -(-blocks // longest))
    return count, blocks // count * block_size


async def metrics(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(f"{url}/metrics") as response:
        return await response.text()


async def engine(session: aiohttp.ClientSession, url: str) -> tuple[str, int, int, int]:
    """The engine's model and context, from /v1/models, and its pool's blocks and block size, from its metrics."""
    async with session.get(f"{url}/v1/models") as response:
        card = (await response.json())["data"][0]
    exposition = await metrics(session, url)
    labels = next(
        (labels for _, labels, _ in stats.metric_samples(exposition, (CACHE_CONFIG,))), None
    )
    if labels is None:
        raise RuntimeError(f"{url}/metrics has no {CACHE_CONFIG}")
    return (
        card["id"],
        int(card["max_model_len"]),
        int(labels["num_gpu_blocks"]),
        int(labels["block_size"]),
    )


async def engine_state(session: aiohttp.ClientSession, url: str, num_blocks: int) -> dict[str, int]:
    """The engine's running and waiting requests and free KV blocks, from its metrics."""
    exposition = await metrics(session, url)
    usage = sample(exposition, "vllm:kv_cache_usage_perc")
    return {
        "running": round(sample(exposition, "vllm:num_requests_running")),
        "waiting": round(sample(exposition, "vllm:num_requests_waiting")),
        "free": round((1 - usage) * num_blocks),
    }


def sample(exposition: str, name: str) -> float:
    """A metric's value in a Prometheus exposition, summed over its label sets."""
    return sum(value for _, _, value in stats.metric_samples(exposition, (name,)))


async def stream(
    session: aiohttp.ClientSession,
    url: str,
    path: str,
    body: dict,
    marks: tuple[int, ...] = (),
    first: asyncio.Event | None = None,
) -> list[float | None]:
    """Seconds to the first streamed token and to each token count in ``marks``, by the usage of each chunk. With
    ``first``, set it at the first token and read the stream to its end: closing the response would make vLLM abort
    the request. Without it, return at the last mark."""
    started = time.perf_counter()
    times: list[float | None] = [None] * (1 + len(marks))
    body = body | {
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    async with session.post(f"{url}{path}", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"{response.status}: {await response.text()}")
        async for line in response.content:
            if not line.startswith(b"data: {"):
                continue
            chunk = json.loads(line[6:])
            tokens = (chunk.get("usage") or {}).get("completion_tokens", 0)
            if times[0] is None and tokens:
                times[0] = time.perf_counter() - started
                if first is not None:
                    first.set()
            for i, mark in enumerate(marks, 1):
                if times[i] is None and tokens >= mark:
                    times[i] = time.perf_counter() - started
            if first is None and marks and times[-1] is not None:
                return times
    if times[0] is None:
        raise RuntimeError("the stream ended before its first token")
    return times


def connect(address: str) -> tuple[aiohttp.ClientSession, str]:
    """A session to the engine at ``address``, and the base of its requests' URLs: on the engine's socket, they
    reach it whatever their host."""
    socket = address.removeprefix("unix:") if address.startswith("unix:") else None
    connector = aiohttp.UnixConnector(path=socket) if socket else None
    session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None))
    return session, "http://localhost" if socket else address


async def main(address: str, case: str, reply: int | None) -> None:
    rng = random.Random()
    session, url = connect(address)
    async with session:
        model, max_model_len, num_gpu_blocks, block_size = await engine(session, url)
        num_blocks = num_gpu_blocks - 1  # less vLLM's null block
        count, prompt_tokens = background(case, num_gpu_blocks, block_size, max_model_len)
        bodies = [
            {
                "model": model,
                "prompt": [rng.randrange(1000, 100_000) for _ in range(prompt_tokens)],
                "max_tokens": DECODE_TOKENS,
                "ignore_eos": True,
                "priority": 2,
            }
            for _ in range(count)
        ]
        started = [asyncio.Event() for _ in bodies]
        background_tasks = [
            asyncio.create_task(stream(session, url, "/v1/completions", body, first=event))
            for body, event in zip(bodies, started, strict=True)
        ]
        if case == "kv":
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 120)
            if not all(event.is_set() for event in started):
                sys.exit("probe: the priority-2 requests did not start decoding within 120 s")

        # A voice session's history that the engine has never seen, so nothing of it is cached,
        # ending on the caller's turn.
        words = CALL.split()
        system = "You are a voice agent on a phone call. Answer briefly."
        messages = [{"role": "system", "content": system}] + [
            {"role": role, "content": " ".join(rng.choices(words, k=120)) + f" ({rng.random()})"}
            for role in ("user", "assistant", "user", "assistant", "user")
        ]
        tokenize = {"model": model, "messages": messages}
        async with session.post(f"{url}/tokenize", json=tokenize) as response:
            voice_tokens = (await response.json())["count"]

        # kv: send the voice turn while the priority-2 requests all run and the pool is full: the engine
        # reports it, so the probe does not rely on timing. Once they have grown into the free
        # blocks, vLLM moves one of them back to waiting, and a voice turn would find room.
        # prefill: send it once all of them run, a few chunks into their prompts.
        while True:
            state = await engine_state(session, url, num_blocks)
            if state["running"] == count and (case == "prefill" or state["free"] <= FREE_BLOCKS):
                break
            if case == "kv" and state["running"] < count:
                sys.exit(
                    f"probe: a priority-2 request stopped running before the pool filled ({state})"
                )
            await asyncio.sleep(0.05)
        body = {"model": model, "messages": messages, "priority": 0}
        if reply is None:
            marks, body["max_tokens"] = (FIRST_TOKENS,), 2 * FIRST_TOKENS
        else:
            marks, body["max_tokens"], body["ignore_eos"] = (FIRST_TOKENS, reply), reply, True
        ttft, *marked = await stream(session, url, "/v1/chat/completions", body, marks)
        after = await engine_state(session, url, num_blocks)
        timed = ", ".join(
            f"{mark} tokens " + ("—" if seconds is None else f"{seconds * 1000:.0f} ms")
            for mark, seconds in zip(marks, marked, strict=True)
        )
        print(
            f"{case}: priority-0 TTFT {ttft * 1000:.0f} ms, {timed},"
            f" for {voice_tokens} prompt tokens, sent with {state['running']} priority-2 requests of"
            f" {prompt_tokens} tokens running and {state['free']} of {num_blocks} blocks free; after"
            f" it, {after['running']} running and {after['waiting']} waiting"
        )
        if case == "prefill" and all(event.is_set() for event in started):
            print("probe: a priority-2 request had its first token before the voice turn ended")
        for task in background_tasks:
            task.cancel()
        await asyncio.gather(*background_tasks, return_exceptions=True)


def reply_tokens(value: str) -> int:
    length = int(value)
    if length <= FIRST_TOKENS:
        raise argparse.ArgumentTypeError(f"a whole reply is longer than {FIRST_TOKENS} tokens")
    return length


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "engine",
        nargs="?",
        default=ENGINE,
        help="the engine's address: unix: and its socket (the GPU node's), or http://host:port",
    )
    parser.add_argument("case", nargs="?", default="kv", choices=("kv", "prefill"))
    parser.add_argument(
        "--reply", type=reply_tokens, metavar="TOKENS", help="the voice turn's whole reply"
    )
    args = parser.parse_args()
    asyncio.run(main(args.engine, args.case, args.reply))
