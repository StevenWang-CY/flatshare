"""Drive a vLLM OpenAI server with the shape of load under which its FP8 KV cache crashed beside CPU offloading.

Coding-agent sessions resend a long context that grows by a little each turn, then pause, so the GPU pool overflows
into the CPU tier and resumed sessions load their prefixes back; two voice-like streams send short high-priority turns
every few seconds. Prompts are token ids (/v1/completions), so lengths and shared prefixes are exact, and every
choice comes from a seeded generator, so a run can be repeated.

usage: python load.py (--uds PATH | --url URL) --duration S --out DIR [--agents N] [--voice N] [--seed N] [--stall S]
Writes DIR/requests.jsonl (one line per request) and DIR/summary.json; exits 3 when the server died during the run, and
4 when it stalled: requests in flight and none finished for --stall seconds, while /health still answered.
"""

import argparse
import asyncio
import contextlib
import json
import random
import time
from pathlib import Path

import httpx

VOCAB = (100, 150_000)  # ordinary Qwen3 token ids, clear of the special tokens at 151,643 and above


class Run:
    def __init__(self, args, client, log):
        self.args = args
        self.client = client
        self.t0 = time.monotonic()
        self.dead_at = None
        self.stalled_at = None
        self.inflight = 0
        self.last_done = time.monotonic()
        self.stop = asyncio.Event()
        self.log = log
        self.counts = {}

    def elapsed(self):
        return time.monotonic() - self.t0

    def record(self, **row):
        row["t"] = round(self.elapsed(), 3)
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()
        key = f"{row['kind']}:{row['status']}"
        self.counts[key] = self.counts.get(key, 0) + 1

    async def complete(self, kind, session, prompt, max_tokens, priority, stream):
        body = {
            "model": self.args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "priority": priority,
            "stream": stream,
        }
        start = self.elapsed()
        first = None
        self.inflight += 1
        try:
            if stream:
                async with self.client.stream("POST", "/v1/completions", json=body) as r:
                    status = r.status_code
                    async for line in r.aiter_lines():
                        if line.startswith("data: ") and first is None and line != "data: [DONE]":
                            first = self.elapsed() - start
                    if status != 200:
                        await r.aread()
            else:
                r = await self.client.post("/v1/completions", json=body)
                status = r.status_code
            error = None if status == 200 else f"http {status}"
        except (httpx.HTTPError, OSError) as e:
            status, error = "error", f"{type(e).__name__}: {e}"[:200]
        finally:
            self.inflight -= 1
            self.last_done = time.monotonic()
        self.record(
            kind=kind,
            session=session,
            prompt_tokens=len(prompt),
            max_tokens=max_tokens,
            status=status,
            error=error,
            seconds=round(self.elapsed() - start, 3),
            ttft=None if first is None else round(first, 3),
        )
        return status == 200

    async def sleep(self, seconds):
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)

    async def agent(self, i):
        """One coding-agent session: a long context, grown by a tool result and a reply each turn, then a pause."""
        rng = random.Random(self.args.seed * 1000 + i)
        await self.sleep(rng.uniform(0, self.args.ramp))
        while not self.stop.is_set():
            context = [rng.randrange(*VOCAB) for _ in range(rng.randint(8_000, 14_000))]
            while not self.stop.is_set() and len(context) < 24_000:
                await self.complete("agent", i, context, rng.randint(40, 400), 1, False)
                context = context + [rng.randrange(*VOCAB) for _ in range(rng.randint(300, 1_500))]
                pause = rng.uniform(20, 60) if rng.random() < 0.2 else rng.uniform(1, 20)
                await self.sleep(pause)

    async def voice(self, i):
        """One call: a short conversation, a turn every few seconds at the highest priority, streamed."""
        rng = random.Random(self.args.seed * 1000 + 500 + i)
        await self.sleep(self.args.ramp + rng.uniform(0, 5))
        while not self.stop.is_set():
            conversation = [rng.randrange(*VOCAB) for _ in range(1_200)]
            for _ in range(25):
                if self.stop.is_set():
                    break
                conversation = conversation + [
                    rng.randrange(*VOCAB) for _ in range(rng.randint(20, 80))
                ]
                await self.complete("voice", i, conversation, 48, 0, True)
                conversation = conversation + [rng.randrange(*VOCAB) for _ in range(48)]
                await self.sleep(rng.uniform(2, 5))

    async def watch(self):
        """End the run at its duration, as soon as the server stops answering /health, or when it stalls: requests in
        flight and none finished for --stall seconds."""
        misses = 0
        while self.elapsed() < self.args.duration:
            await self.sleep(2)
            try:
                ok = (await self.client.get("/health", timeout=10)).status_code == 200
            except (httpx.HTTPError, OSError):
                ok = False
            misses = 0 if ok else misses + 1
            if misses >= 3:
                self.dead_at = round(self.elapsed(), 1)
                break
            if self.inflight and time.monotonic() - self.last_done > self.args.stall:
                self.stalled_at = round(self.elapsed(), 1)
                break
        self.stop.set()


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uds")
    p.add_argument("--url", default="http://localhost")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507-FP8")
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--agents", type=int, default=24)
    p.add_argument("--voice", type=int, default=2)
    p.add_argument("--ramp", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--stall", type=float, default=180.0)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    transport = httpx.AsyncHTTPTransport(uds=args.uds) if args.uds else None
    limits = httpx.Limits(max_connections=args.agents + args.voice + 4)
    client = httpx.AsyncClient(base_url=args.url, transport=transport, timeout=600, limits=limits)
    with open(Path(args.out) / "requests.jsonl", "w") as log:
        async with client:
            run = Run(args, client, log)
            tasks = [asyncio.create_task(run.agent(i)) for i in range(args.agents)]
            tasks += [asyncio.create_task(run.voice(i)) for i in range(args.voice)]
            await run.watch()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    summary = {
        "seconds": round(run.elapsed(), 1),
        "server_died_at": run.dead_at,
        "stalled_at": run.stalled_at,
        "counts": run.counts,
        "args": vars(args),
    }
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    if run.dead_at is not None:
        return 3
    return 4 if run.stalled_at is not None else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
