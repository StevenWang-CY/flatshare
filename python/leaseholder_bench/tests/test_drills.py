"""The failure drills' client: how a stream's end is classified, the recovery probe, and the summary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from leaseholder_bench.cli import main
from leaseholder_bench.drills import (
    BROKEN,
    DONE,
    ERROR_EVENT,
    HTTP_ERROR,
    DrillConfig,
    StreamRecord,
    fill_cpu_tier,
    probe,
    read_stream,
    report,
    run_streams,
    start_phases,
)

CHUNK = b'data: {"choices": [{"delta": {"content": "1\\n"}}]}\n\n'


class Body(httpx.AsyncByteStream):
    """A streamed body of ``chunks`` that then ends, or breaks off as a connection that closes mid-body does."""

    def __init__(self, chunks: list[bytes], *, broken: bool = False) -> None:
        self.chunks, self.broken = chunks, broken

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.broken:
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://gateway")


async def one(response: httpx.Response) -> StreamRecord:
    record = StreamRecord(0, "realtime", 0.0)
    async with client(lambda request: response) as c:
        await read_stream(c, "/v1/chat/completions", {}, {}, record)
    return record


def test_a_stream_that_ends_with_done_is_done() -> None:
    record = asyncio.run(one(httpx.Response(200, stream=Body([CHUNK, CHUNK, b"data: [DONE]\n\n"]))))
    assert (record.status, record.end, record.chunks, record.error) == (200, DONE, 2, None)
    assert record.first_byte is not None and record.ended >= record.last_chunk >= record.first_byte


def test_a_stream_whose_connection_breaks_off_is_broken_with_the_transport_error() -> None:
    record = asyncio.run(one(httpx.Response(200, stream=Body([CHUNK], broken=True))))
    assert (record.end, record.chunks) == (BROKEN, 1)
    assert record.error.startswith("RemoteProtocolError: peer closed connection")


def test_a_stream_that_ends_without_done_is_broken() -> None:
    record = asyncio.run(one(httpx.Response(200, stream=Body([CHUNK]))))
    assert (record.end, record.error) == (BROKEN, "the stream ended without [DONE]")


def test_an_error_event_ends_the_stream_with_its_type() -> None:
    event = (
        b'data: {"error": {"type": "upstream_unavailable", "message": "the engine went away"}}\n\n'
    )
    record = asyncio.run(one(httpx.Response(200, stream=Body([CHUNK, event]))))
    assert (record.end, record.chunks) == (ERROR_EVENT, 1)
    assert record.error == "upstream_unavailable: the engine went away"


def test_an_error_answer_is_an_http_error_with_its_type() -> None:
    body = {
        "error": {"type": "upstream_unavailable", "message": "upstream request failed: connect"}
    }
    record = asyncio.run(one(httpx.Response(502, json=body)))
    assert (record.status, record.end, record.first_byte) == (502, HTTP_ERROR, None)
    assert record.error == "upstream_unavailable: upstream request failed: connect"


def test_the_probe_retries_until_a_request_succeeds() -> None:
    answers = iter(
        [
            httpx.ConnectError("refused"),
            httpx.Response(
                502, json={"error": {"type": "upstream_unavailable", "message": "down"}}
            ),
            httpx.Response(200, json={"choices": []}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def go():
        async with client(handler) as c:
            return await probe(c, "/v1/chat/completions", {}, DrillConfig(probe_interval_s=0.0))

    found = asyncio.run(go())
    assert [outcome for _, outcome in found.attempts] == [
        "ConnectError",
        "http 502 upstream_unavailable",
        "ok",
    ]
    assert found.recovered == found.attempts[-1][0]


def test_the_drill_records_each_class_and_the_probe(tmp_path: Path, capsys) -> None:
    seen: list[tuple[str, dict]] = []

    def voice(request: httpx.Request) -> httpx.Response:
        seen.append(("voice", json.loads(request.content)))
        return httpx.Response(200, stream=Body([CHUNK, b"data: [DONE]\n\n"]))

    def agents(request: httpx.Request) -> httpx.Response:
        seen.append(("agents", json.loads(request.content)))
        assert request.headers["authorization"] == "Bearer k"
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, stream=Body([CHUNK], broken=True))
        return httpx.Response(200, json={"choices": []})

    config = DrillConfig(realtime=1, background=2, max_tokens=5, probe_interval_s=0.0)
    record = asyncio.run(
        run_streams(config, client(voice), client(agents), "k", tmp_path / "drill.json")
    )
    assert [s["end"] for s in record["streams"]] == [DONE, BROKEN, BROKEN]
    assert [s["slo_class"] for s in record["streams"]] == ["realtime", "background", "background"]
    assert record["probe"]["recovered"] is not None
    assert json.loads((tmp_path / "drill.json").read_text()) == json.loads(json.dumps(record))
    assert all(body["max_tokens"] == 5 and body["stream"] for who, body in seen[:3])
    assert "streaming" in capsys.readouterr().out


def test_the_report_counts_the_ends_and_times_them_from_the_event(tmp_path: Path, capsys) -> None:
    record = {
        "streams": [
            {
                "slo_class": "realtime",
                "end": BROKEN,
                "ended": 101.5,
                "error": "RemoteProtocolError: closed",
            },
            {
                "slo_class": "background",
                "end": BROKEN,
                "ended": 100.2,
                "error": "RemoteProtocolError: closed",
            },
            {"slo_class": "background", "end": DONE, "ended": 90.0, "error": None},
        ],
        "probe": {
            "attempts": [[100.5, "http 502 upstream_unavailable"], [160.0, "ok"]],
            "recovered": 160.0,
        },
    }
    text = report(record, 100.0, "the kill")
    assert "realtime: 1 streams: 1 broken" in text
    assert "background: 2 streams: 1 broken, 1 done" in text
    assert "ended 1,500 ms after the kill at the median" in text
    assert (
        "probe: 2 attempts (1 http 502 upstream_unavailable); the first success 60.0 s after the kill"
        in text
    )
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    assert main(["drill", "report", str(path), "--event", "100", "--label", "the kill"]) == 0
    assert capsys.readouterr().out.strip() == text


@pytest.mark.parametrize("realtime", [0, 1])
def test_streams_of_one_class_only_are_reported(realtime: int) -> None:
    record = {
        "streams": [
            {
                "slo_class": "realtime" if realtime else "background",
                "end": DONE,
                "ended": 1.0,
                "error": None,
            }
        ],
        "probe": {"attempts": [[2.0, "ok"]], "recovered": 2.0},
    }
    assert report(record, None).count("streams:") == 1


ENGINE_START = """\
1792890463.426422 phastform systemd[2135572]: Stopped Leaseholder LLM engine.
1792890463.682588 phastform systemd[2135572]: Starting Leaseholder LLM engine: vLLM 0.32...
1792890480.031740 phastform vllm[3777953]: (APIServer pid=3777953) INFO 10-24 21:08:00 [api_utils.py:347]  version 0.32.0
1792890501.908518 phastform vllm[3778158]: (EngineCore pid=3778158) INFO 10-24 21:08:21 [core.py:133] Initializing a V1 LLM engine (v0.32.0) with config: ...
1792890505.338407 phastform vllm[3778158]: (EngineCore pid=3778158) INFO 10-24 21:08:25 [flash_attn.py:1117] Using FlashAttention version 2
1792890517.593438 phastform vllm[3778158]: (EngineCore pid=3778158) INFO 10-24 21:08:37 [model_runner.py:424] Model loading took 4.29 GiB memory and 14.314462 seconds
1792890518.900137 phastform vllm[3778158]: (EngineCore pid=3778158) INFO 10-24 21:08:38 [kv_cache_utils.py:2327] GPU KV cache size: 36,400 tokens
1792890547.480065 phastform vllm[3778158]: (EngineCore pid=3778158) INFO 10-24 21:09:07 [model_runner.py:1127] Graph capturing finished in 12 secs, took 0.42 GiB
1792890549.316518 phastform vllm[3778158]: (EngineCore pid=3778158) INFO 10-24 21:09:09 [core.py:395] init engine (profile, create kv cache, warmup model) took 31.68 s (compilation: 0.32 s)
1792890551.255948 phastform vllm[3777953]: (APIServer pid=3777953) INFO 10-24 21:09:11 [entry.py:140] Starting vLLM server on http://127.0.0.1:8000
1792890553.101000 phastform systemd[2135572]: Started Leaseholder LLM engine: vLLM 0.32.
"""


def test_a_start_is_timed_by_its_phases_from_the_last_starting_line(tmp_path: Path, capsys) -> None:
    phases = dict(
        start_phases(
            "1792890000.0 phastform systemd[1]: Starting an earlier start\n" + ENGINE_START
        )
    )
    assert phases == {
        "the unit starts": 0.0,
        "the API server is up": 16.3,
        "the engine process is up": 38.2,
        "the weights are loaded": 53.9,
        "the KV cache is allocated": 55.2,
        "the CUDA graphs are captured": 83.8,
        "the engine is initialized": 85.6,
        "the server listens": 87.6,
        "the unit is ready (its health check)": 89.4,
    }
    assert start_phases("no journal lines\n") == []
    path = tmp_path / "engine-journal.txt"
    path.write_text(ENGINE_START)
    assert main(["drill", "start", str(path)]) == 0
    assert "     87.6 s  the server listens" in capsys.readouterr().out


class FakeEngine:
    """An engine with a GPU pool of ``pool`` tokens and a CPU tier of ``tier_bytes``, both least recently
    used first, at ``kv`` bytes a token, whose /metrics count the hits and the tier's stores."""

    def __init__(self, pool: int = 1000, tier_bytes: int = 10_000, kv: int = 2) -> None:
        self.pool, self.tier_bytes, self.kv = pool, tier_bytes, kv
        self.gpu: dict[tuple, int] = {}
        self.cpu: dict[tuple, int] = {}
        self.gpu_hits = self.cpu_hits = self.stored = 0

    @staticmethod
    def _keep(cache: dict[tuple, int], key: tuple, size: int, limit: int) -> None:
        cache.pop(key, None)
        cache[key] = size
        while sum(cache.values()) > limit:
            cache.pop(next(iter(cache)))

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/server_info":
            extra = {"cpu_bytes_to_use": self.tier_bytes}
            return httpx.Response(
                200,
                json={"vllm_config": {"kv_transfer_config": {"kv_connector_extra_config": extra}}},
            )
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/metrics":
            text = (
                f'vllm:cache_config_info{{block_size="1",num_gpu_blocks="{self.pool}"}} 1\n'
                f"vllm:prefix_cache_hits_total {self.gpu_hits}\n"
                f"vllm:external_prefix_cache_hits_total {self.cpu_hits}\n"
                f"vllm:kv_offload_store_bytes_total {self.stored}\n"
            )
            return httpx.Response(200, text=text)
        prompt = tuple(json.loads(request.content)["prompt"])
        if prompt in self.gpu:
            self.gpu_hits += len(prompt)
        elif prompt in self.cpu:
            self.cpu_hits += len(prompt)
        self._keep(self.gpu, prompt, len(prompt), self.pool)
        self._keep(self.cpu, prompt, len(prompt) * self.kv, self.tier_bytes)
        self.stored += len(prompt) * self.kv
        return httpx.Response(200, json={"usage": {"prompt_tokens": len(prompt)}})


def test_a_full_cpu_tier_evicts_the_oldest_prompt_and_keeps_the_latest(tmp_path: Path) -> None:
    engine = FakeEngine(pool=20_000, tier_bytes=100_000, kv=2)  # the tier holds 50,000 tokens
    record = asyncio.run(
        fill_cpu_tier(client(engine.handler), tmp_path / "tier.json", factor=2.0, tokens=1000)
    )
    assert record["passed"] and record["healthy"]
    assert record["stored_bytes"] >= 2 * 100_000
    tiers = [(s["tier"], s["cpu_hit_tokens"], s["gpu_hit_tokens"]) for s in record["samples"]]
    assert tiers == [
        ("recompute", 0, 0),
        ("recompute", 0, 0),
        ("cpu", 1000, 0),
        ("recompute", 0, 0),
    ]
    assert (
        json.loads((tmp_path / "tier.json").read_text())["fill_requests"] == record["fill_requests"]
    )


class StalledTier(FakeEngine):
    """An engine whose CPU tier stores nothing, so its store counter never moves."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        stored = self.stored
        response = super().handler(request)
        self.stored = stored
        return response


def test_the_fills_stop_when_the_cpu_tier_stores_nothing(tmp_path: Path) -> None:
    record = asyncio.run(
        fill_cpu_tier(client(StalledTier().handler), tmp_path / "tier.json", tokens=1000)
    )
    assert (record["passed"], record["fill_requests"]) == (False, 3)
    assert record["failure"] == "the CPU tier's store counter did not move in 3 fills"
    assert [s["tier"] for s in record["samples"]] == ["recompute"]


def test_the_fills_stop_at_twice_what_the_tiers_rate_needs(tmp_path: Path) -> None:
    class Slowing(FakeEngine):
        """A tier that stores its first three fills, then nothing."""

        def handler(self, request: httpx.Request) -> httpx.Response:
            stored = self.stored
            response = super().handler(request)
            if json.loads(request.content or b"{}").get("prompt") and self.fills >= 4:
                self.stored = stored
            self.fills += 1 if json.loads(request.content or b"{}").get("prompt") else 0
            return response

    engine = Slowing(pool=20_000, tier_bytes=1_000_000, kv=2)
    engine.fills = 0
    record = asyncio.run(fill_cpu_tier(client(engine.handler), tmp_path / "tier.json", tokens=1000))
    assert record["passed"] is False and record["failure"].startswith("the CPU tier stored ")
    # Three fills store 98,304 bytes against a target of 2,000,000: the limit is 2 * 62 + 2 fills.
    assert record["fill_requests"] == 2 * 62 + 2
