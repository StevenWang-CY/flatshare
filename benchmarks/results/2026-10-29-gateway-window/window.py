"""rv3_window.py: RV3 lane P's measurement in the GPU window on the shipped v1.7 serving profile, from cyan.

Run as the deploy user on cyan, with the bench venv and nothing else, after v1.7 is deployed and before Phastform is
paused, with no call in progress:

    timeout 1500 ~/flatshare/venvs/bench/bin/python rv3_window.py ~/flatshare/runs/rv3-$(date -u +%Y%m%dT%H%MZ)

It reads and writes nothing but the sockets below and its output directory (created; it refuses one that exists or
lies outside ~/flatshare/runs/rv3-*), changes no unit, configuration or release, and sends only the voice agent's own
kind of requests, one at a time: 164 short LLM requests (82 through the gateway, all as one realtime session) and 126
TTS requests, about 4 minutes on the box (each part has its own time bound; the whole run stops at 20 minutes). It
exits 2 without measuring if a socket is missing, the gateway is not ready, a call is on (a realtime request in flight,
a realtime stream active, or a realtime session placed on a node: after a call ends, or after a run of this script,
wait for the session to idle out, the gateway's placement idle_s), or the output directory exists; 0 when it has
written results.json; 1 when it has written it but a part failed (results.json's "errors" says which and why).

What it measures (each part paired or interleaved, so that drift hits both sides alike):
  A. The gateway's added latency on the real path, v1.7 against W6's record (benchmarks/results/2026-10-25-w6-transport,
     a 0.29 ms paired median on the first token at v0.3-191): the voice agent's LLM request (greedy, 60 tokens) once
     on each path per round, alternating, 80 rounds after 2 warm-up rounds: "direct" is flatshare-tunnel's engine
     socket, "gateway" the gateway's voice socket, whose upstream is that socket. Per request: the first content token,
     the first sentence (20 characters at least, the worker's cut), the end, and the gaps between content chunks.
  B. The gateway's own histogram over part A (flatshare_gateway_overhead_seconds and its first-byte counters on the
     voice socket's /metrics, which needs no key), before and after.
  C. TTS first audio against the text's length on the GPU: Kokoro on the tunnel's TTS socket as the worker asks for it
     (/v1/audio/speech, af_heart, raw PCM, streamed), three lengths interleaved, 40 requests each after 2 warm-ups: a
     clause (the first words up to a comma), the voice agent's typical first sentence, and two sentences. Per request:
     the first audio bytes, the last, and the audio's length.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx

# The deploy user's runtime directory on cyan; RV3_RUN_DIR and RV3_RUNS_DIR only for a dry run elsewhere.
RUN = Path(os.environ.get("RV3_RUN_DIR", f"/run/user/{os.getuid()}/flatshare"))
RUNS = Path(os.environ.get("RV3_RUNS_DIR", Path.home() / "flatshare" / "runs"))
SOCKETS = {
    "engine": RUN / "phastform-engine.sock",
    "gateway": RUN / "gateway.sock",
    "tts": RUN / "phastform-tts.sock",
}
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
TEXTS = {
    "clause": "Yes, we do,",
    "sentence": "Yes, we have a table for four at seven tomorrow.",
    "two_sentences": "Yes, we have a table for four at seven tomorrow. Would you like me to book it under your name?",
}
SESSION = "voice-rv3-probe"
ROUNDS_A, WARMUP_A, ROUNDS_C, WARMUP_C = 80, 2, 40, 2
BOUND_A_S, BOUND_C_S, BOUND_ALL_S = 600, 300, 1200
STARTED = time.monotonic()


try:  # the bench venv's own rule, the voice worker's cut (W6's llm_paths.py imports it too)
    from flatshare_bench.voice_turn import first_sentence
except ImportError:  # pragma: no cover - only without flatshare_bench

    def first_sentence(text: str) -> str | None:
        """The text up to the first sentence end that leaves at least 20 characters."""
        for i, ch in enumerate(text):
            if ch in ".!?" and len(text[: i + 1].strip()) >= 20:
                return text[: i + 1].strip()
        return None


def client(socket: Path) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=str(socket)), base_url="http://local", timeout=60
    )


async def llm(c: httpx.AsyncClient, path: str, line: str, n: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": line}],
        "max_tokens": 60,
        "temperature": 0,
        "stream": True,
    }
    # One session for every gateway request: with [[nodes]], each new realtime session takes one of a node's call
    # slots (max_calls) until it idles out, so per-round sessions would fill them.
    headers = {"X-Flatshare-Class": "realtime", "X-Session-ID": SESSION}
    text, stamps, first_s, sentence_s = "", [], None, None
    start = time.perf_counter()
    async with c.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
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
            if first_s is None:
                first_s = now
            if sentence_s is None and first_sentence(text) is not None:
                sentence_s = now
    end_s = time.perf_counter() - start
    gaps = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    return {
        "path": path,
        "n": n,
        "first_ms": None if first_s is None else first_s * 1e3,
        "sentence_ms": None if sentence_s is None else sentence_s * 1e3,
        "end_ms": end_s * 1e3,
        "chunks": len(stamps),
        "gaps_ms": [round(g, 3) for g in gaps],
        "text": text,
    }


async def tts(c: httpx.AsyncClient, kind: str, n: int) -> dict:
    body = {"model": "kokoro", "input": TEXTS[kind], "voice": "af_heart", "response_format": "pcm", "stream": True}
    first_s, size = None, 0
    start = time.perf_counter()
    async with c.stream("POST", "/v1/audio/speech", json=body, timeout=60) as r:
        r.raise_for_status()
        async for chunk in r.aiter_bytes():
            if chunk and first_s is None:
                first_s = time.perf_counter() - start
            size += len(chunk)
    total_s = time.perf_counter() - start
    return {
        "kind": kind,
        "n": n,
        "first_audio_ms": None if first_s is None else first_s * 1e3,
        "total_ms": total_s * 1e3,
        "audio_s": size / 2 / 24000,  # 16-bit mono at 24 kHz
    }


async def scrape(c: httpx.AsyncClient) -> str:
    r = await c.get("/metrics")
    r.raise_for_status()
    return r.text


def series(text: str, prefix: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if line.startswith(prefix):
            name, _, value = line.rpartition(" ")
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


def summary(values: list[float]) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    q = statistics.quantiles(values, n=20, method="inclusive") if len(values) > 1 else [values[0]] * 19
    return {"n": len(values), "p50": statistics.median(values), "p90": q[17], "p95": q[18], "min": values[0], "max": values[-1]}


def late(bound: float, since: float) -> bool:
    return time.monotonic() - since > bound or time.monotonic() - STARTED > BOUND_ALL_S


async def main(out: Path) -> int:
    missing = [name for name, path in SOCKETS.items() if not path.is_socket()]
    if missing:
        print(f"rv3_window: no socket for {', '.join(missing)} under {RUN}: not measuring", file=sys.stderr)
        return 2
    gw, engine, tts_c = client(SOCKETS["gateway"]), client(SOCKETS["engine"]), client(SOCKETS["tts"])
    try:
        ready = await gw.get("/ready")
        if ready.status_code != 200:
            print(f"rv3_window: the gateway is not ready ({ready.status_code} {ready.text!r}): not measuring", file=sys.stderr)
            return 2
        before = await scrape(gw)
        busy = {
            k: v
            for name in ("flatshare_gateway_in_flight", "flatshare_gateway_node_sessions", "flatshare_gateway_streams_active")
            for k, v in series(before, name).items()
            if 'class="realtime"' in k and v > 0
        }
        if busy:
            print(f"rv3_window: realtime requests in flight ({busy}): a call is on, not measuring", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=False)
        record: dict = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "python": platform.python_version(),
            "httpx": httpx.__version__,
            "sockets": {k: str(v) for k, v in SOCKETS.items()},
        }
        record["errors"] = []
        # A: the paired LLM probe, the paths alternating in order from round to round.
        rows, since = [], time.monotonic()
        paths = {"direct": engine, "gateway": gw}
        try:
            for n in range(WARMUP_A + ROUNDS_A):
                if late(BOUND_A_S, since):
                    record["a_stopped_early_at_round"] = n
                    break
                order = ["direct", "gateway"] if n % 2 == 0 else ["gateway", "direct"]
                line = LINES[n % len(LINES)]
                for path in order:
                    row = await llm(paths[path], path, line, n)
                    row["warmup"] = n < WARMUP_A
                    rows.append(row)
        except Exception as error:  # a failed request ends part A; C still runs and results.json says why
            record["errors"].append(f"A: {type(error).__name__}: {error}")
        after = await scrape(gw)
        measured = [r for r in rows if not r["warmup"]]
        by = {p: [r for r in measured if r["path"] == p] for p in paths}
        paired = {}
        for key in ("first_ms", "sentence_ms", "end_ms"):
            diffs = []
            for n in sorted({r["n"] for r in measured}):
                pair = {r["path"]: r for r in measured if r["n"] == n}
                if len(pair) == 2 and pair["direct"][key] is not None and pair["gateway"][key] is not None:
                    diffs.append(pair["gateway"][key] - pair["direct"][key])
            paired[key] = summary(diffs)
        record["a"] = {
            "summary": {p: {k: summary([r[k] for r in rs]) for k in ("first_ms", "sentence_ms", "end_ms")}
                        | {"gaps_ms": summary([g for r in rs for g in r["gaps_ms"]])} for p, rs in by.items()},
            "paired_gateway_minus_direct_ms": paired,
            "same_text": all(len({r["text"] for r in measured if r["n"] == n}) == 1 for n in {r["n"] for r in measured}),
            "rows": rows,
        }
        # B: the gateway's own histogram over A.
        names = (
            "flatshare_gateway_overhead_seconds",
            "flatshare_gateway_request_body_seconds",
            "flatshare_gateway_first_byte_seconds",
        )
        record["b"] = {
            "before": {k: v for name in names for k, v in series(before, name).items()},
            "after": {k: v for name in names for k, v in series(after, name).items()},
        }
        # C: TTS first audio by the text's length, the kinds interleaved.
        trows, since = [], time.monotonic()
        kinds = list(TEXTS)
        try:
            for n in range(WARMUP_C + ROUNDS_C):
                if late(BOUND_C_S, since):
                    record["c_stopped_early_at_round"] = n
                    break
                for kind in kinds[n % 3:] + kinds[: n % 3]:
                    row = await tts(tts_c, kind, n)
                    row["warmup"] = n < WARMUP_C
                    trows.append(row)
        except Exception as error:
            record["errors"].append(f"C: {type(error).__name__}: {error}")
        record["c"] = {
            "texts": TEXTS,
            "summary": {k: {f: summary([r[f] for r in trows if r["kind"] == k and not r["warmup"]])
                            for f in ("first_audio_ms", "total_ms", "audio_s")} for k in kinds},
            "rows": trows,
        }
        record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "results.json").write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps({"a_paired": paired, "c": record["c"]["summary"], "errors": record["errors"]}, indent=1))
        return 1 if record["errors"] else 0
    finally:
        for c in (gw, engine, tts_c):
            await c.aclose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    target = Path(sys.argv[1]).expanduser().resolve()
    runs = RUNS.resolve()
    if target.parent != runs or not target.name.startswith("rv3-"):
        print(f"rv3_window: the output must be ~/flatshare/runs/rv3-*, not {target}", file=sys.stderr)
        sys.exit(2)
    if target.exists():
        print(f"rv3_window: {target} exists; name a new directory", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(main(target)))
