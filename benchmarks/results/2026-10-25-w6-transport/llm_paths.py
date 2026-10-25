"""llm_paths.py OUT.json [ROUNDS]: the voice agent's LLM request through each path to the engine, alternating.

Paths (Unix sockets on cyan): "direct" is lh-tunnel's engine socket; "gateway" is lh-gateway's voice socket, whose
upstream is that socket. (W1 ran it when both went over the tunnel's TCP forwards; the code is the same.) Each round sends the same request once per path, in
alternating order, so drift hits both. Greedy decoding makes both paths generate the same tokens. Times: first
content token, the first sentence as the voice worker cuts it (at least 20 characters), the end of the stream,
and the gaps between content chunks.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

import httpx
from leaseholder_bench.voice_turn import first_sentence

RUN = f"/run/user/{os.getuid()}/lh"
PATHS = {"direct": f"{RUN}/phastform-engine.sock", "gateway": f"{RUN}/gateway.sock"}
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
SYSTEM = (
    "You are Ava, the phone assistant of a small neighborhood restaurant. You take reservations, "
    "answer questions about the menu, opening hours and directions, and keep every answer to one "
    "or two short, natural sentences. You are speaking, so never use lists, markdown or emoji."
)
LINES = [
    "Hi, I'd like to book a table for four tomorrow at seven.",
    "Do you have anything vegetarian on the menu?",
    "What time do you close on Sundays?",
    "Is there parking near the restaurant?",
    "Can I bring a birthday cake for my daughter?",
]


async def one(client: httpx.AsyncClient, path: str, line: str, n: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": line}],
        "max_tokens": 60,
        "temperature": 0,
        "stream": True,
    }
    headers = {"X-LH-Class": "realtime", "X-Session-ID": f"voice-probe-{path}-{n}"}
    text, stamps = "", []
    sentence_s = first_s = None
    start = time.perf_counter()
    async with client.stream("POST", "http://local/v1/chat/completions", json=body, headers=headers) as r:
        r.raise_for_status()
        async for raw in r.aiter_lines():
            if not raw.startswith("data: ") or raw == "data: [DONE]":
                continue
            chunk = json.loads(raw[6:])
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if not delta:
                continue
            now = time.perf_counter() - start
            stamps.append(now)
            text += delta
            first_s = first_s if first_s is not None else now
            if sentence_s is None and first_sentence(text):
                sentence_s = now
    end_s = time.perf_counter() - start
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"path": path, "n": n, "first_ms": first_s * 1e3, "sentence_ms": (sentence_s or end_s) * 1e3,
            "end_ms": end_s * 1e3, "chunks": len(stamps), "gaps_ms": [g * 1e3 for g in gaps], "text": text}


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


async def main() -> None:
    out, rounds = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40
    clients = {p: httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=s), timeout=60) for p, s in PATHS.items()}
    rows = []
    for warm in range(2):
        for p in PATHS:
            await one(clients[p], p, LINES[warm], -1)
    for n in range(rounds):
        order = list(PATHS) if n % 2 == 0 else list(reversed(PATHS))
        for p in order:
            rows.append(await one(clients[p], p, LINES[n % len(LINES)], n))
            await asyncio.sleep(0.3)
    summary = {}
    for p in PATHS:
        mine = [r for r in rows if r["path"] == p]
        gaps = [g for r in mine for g in r["gaps_ms"]]
        summary[p] = {
            "n": len(mine),
            **{f"{k}_p50": round(statistics.median(r[k] for r in mine), 1) for k in ("first_ms", "sentence_ms", "end_ms")},
            **{f"{k}_p90": round(pct([r[k] for r in mine], 0.9), 1) for k in ("first_ms", "sentence_ms", "end_ms")},
            "gap_p50_ms": round(statistics.median(gaps), 2),
            "gap_p99_ms": round(pct(gaps, 0.99), 2),
            "gaps_over_30ms": sum(g > 30 for g in gaps),
            "gaps": len(gaps),
        }
    pairs = {}
    for k in ("first_ms", "sentence_ms", "end_ms"):
        diffs = []
        for n in range(rounds):
            a = next(r for r in rows if r["n"] == n and r["path"] == "gateway")
            b = next(r for r in rows if r["n"] == n and r["path"] == "direct")
            diffs.append(a[k] - b[k])
        pairs[f"{k}_paired_median_gateway_minus_direct"] = round(statistics.median(diffs), 2)
    same_text = all(
        next(r for r in rows if r["n"] == n and r["path"] == "gateway")["text"]
        == next(r for r in rows if r["n"] == n and r["path"] == "direct")["text"]
        for n in range(rounds)
    )
    result = {"summary": summary, "paired": pairs, "same_text": same_text, "rows": rows}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"summary": summary, "paired": pairs, "same_text": same_text}, indent=1))
    for c in clients.values():
        await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())
