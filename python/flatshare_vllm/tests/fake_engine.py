"""A stand-in for vLLM's OpenAI server, enough for the make-room probes (probe.py and tail.py) to run end to end on a
CPU: /v1/models (the model and its context), /metrics (the pool, the running and waiting requests and the KV cache's
use), /tokenize, and completions and chat completions, streamed (with usage on each chunk if asked) or whole, on a
loopback port or, as the GPU node's engine, on a Unix socket (``socket``). A request
runs from its arrival until its last token or until its client goes; it takes a prefill time in proportion to its
prompt, then its tokens at a fixed pace, and holds the KV blocks of its prompt and its tokens so far. A prompt longer
than the context gets vLLM's 400."""

from __future__ import annotations

import asyncio
import itertools
import json
import math
from dataclasses import dataclass, field

from aiohttp import web
from aiohttp.test_utils import TestServer


@dataclass
class FakeEngine:
    num_gpu_blocks: int = 2275  # the frozen box's pool, vLLM's null block included
    block_size: int = 16
    max_model_len: int = 32768
    model: str = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    prefill_s_per_token: float = 20e-6
    token_s: float = 0.002
    # The requests in flight, each with its prompt's tokens and its tokens so far.
    running: dict[int, list[int]] = field(default_factory=dict)
    bodies: list[dict] = field(default_factory=list)  # every completion's body, in arrival order
    rejected: int = 0
    socket: str | None = None  # a Unix socket's path to serve on, in place of a loopback port

    def __post_init__(self) -> None:
        self._ids = itertools.count()
        self._server: TestServer | None = None
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> str:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_post("/tokenize", self.tokenize)
        app.router.add_post("/v1/completions", self.complete)
        app.router.add_post("/v1/chat/completions", self.complete)
        if self.socket:
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            await web.UnixSite(self._runner, self.socket).start()
            return f"unix:{self.socket}"
        self._server = TestServer(app, host="127.0.0.1")
        await self._server.start_server()
        return str(self._server.make_url("")).rstrip("/")

    async def __aexit__(self, *exc_info) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            return
        assert self._server is not None
        await self._server.close()

    async def models(self, _: web.Request) -> web.Response:
        card = {"id": self.model, "object": "model", "max_model_len": self.max_model_len}
        return web.json_response({"object": "list", "data": [card]})

    async def metrics(self, _: web.Request) -> web.Response:
        num_blocks = self.num_gpu_blocks - 1
        used = sum(math.ceil(sum(sizes) / self.block_size) for sizes in self.running.values())
        model = f'engine="0",model_name="{self.model}"'
        info = f'block_size="{self.block_size}",cache_dtype="auto",num_gpu_blocks="{self.num_gpu_blocks}"'
        return web.Response(
            text=f"vllm:cache_config_info{{{info}}} 1.0\n"
            f"vllm:num_requests_running{{{model}}} {float(len(self.running))}\n"
            f"vllm:num_requests_waiting{{{model}}} 0.0\n"
            f"vllm:kv_cache_usage_perc{{{model}}} {min(1.0, used / num_blocks)}\n"
        )

    @staticmethod
    def words(body: dict) -> int:
        return sum(len(message["content"].split()) for message in body["messages"])

    async def tokenize(self, request: web.Request) -> web.Response:
        return web.json_response({"count": self.words(await request.json())})

    async def complete(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.bodies.append(body)
        chat = "messages" in body
        prompt = self.words(body) if chat else len(body["prompt"])
        if prompt > self.max_model_len:
            self.rejected += 1
            message = f"Input length ({prompt}) exceeds model's maximum context length ({self.max_model_len})."
            return web.json_response({"error": {"message": message}}, status=400)
        tokens = min(body.get("max_tokens") or 16, self.max_model_len - prompt)
        request_id = next(self._ids)
        self.running[request_id] = [prompt, 0]
        try:
            if not body.get("stream"):
                await asyncio.sleep(prompt * self.prefill_s_per_token + tokens * self.token_s)
                usage = {"prompt_tokens": prompt, "completion_tokens": tokens}
                return web.json_response(
                    {"choices": [{"index": 0, "text": " t" * tokens}], "usage": usage}
                )
            return await self.stream(request, request_id, body, chat, prompt, tokens)
        finally:
            self.running.pop(request_id, None)

    async def stream(
        self,
        request: web.Request,
        request_id: int,
        body: dict,
        chat: bool,
        prompt: int,
        tokens: int,
    ) -> web.StreamResponse:
        options = body.get("stream_options") or {}
        continuous = bool(options.get("include_usage") and options.get("continuous_usage_stats"))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        def usage(done: int) -> dict:
            return {
                "prompt_tokens": prompt,
                "completion_tokens": done,
                "total_tokens": prompt + done,
            }

        async def send(choices: list[dict], done: int | None) -> None:
            chunk: dict = {"id": f"req-{request_id}", "choices": choices}
            if done is not None:
                chunk["usage"] = usage(done)
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())

        def choice(text: str, finish: str | None = None, role: bool = False) -> dict:
            if not chat:
                return {"index": 0, "text": text, "finish_reason": finish}
            delta = (
                {"role": "assistant", "content": text}
                if role
                else ({"content": text} if text else {})
            )
            return {"index": 0, "delta": delta, "finish_reason": finish}

        try:
            if chat:
                await send([choice("", role=True)], 0 if continuous else None)
            await asyncio.sleep(prompt * self.prefill_s_per_token)
            for done in range(1, tokens + 1):
                if done > 1:
                    await asyncio.sleep(self.token_s)
                self.running[request_id][1] = done
                await send([choice(f" t{done}")], done if continuous else None)
            await send([choice("", finish="length")], tokens if continuous else None)
            if options.get("include_usage"):
                await send([], tokens)
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        except (
            ConnectionResetError
        ):  # the client went, as probe.py's voice turn does at its last mark
            pass
        return response
