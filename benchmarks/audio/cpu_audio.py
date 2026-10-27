"""Speech synthesis and recognition on the CPU: whether the voice servers' GPU memory (about 3.4 GiB, STT 2.4 and TTS
1.0, docs/ARCHITECTURE.md) could go to the KV cache, by how long the first audio and a transcript take on the host's
CPU with 1, 2 and 4 calls at once.

- ``tts``: the box's TTS server, Kokoro-FastAPI, started on the CPU (``USE_GPU=false``) on a socket, asked for each
  sentence as the voice agent asks for it (``/v1/audio/speech``, the agent's voice ``af_heart``, raw 24 kHz PCM,
  streamed): the time to its first audio bytes and to its last, ``c`` requests at once.
- ``stt``: faster-whisper with Whisper large-v3-turbo's weights in INT8 on the CPU, the caller corpus's lines
  transcribed whole, as the worker's STT gets each VAD-cut utterance, ``c`` at once: the time to the transcript, and
  its words against the line's text.

    python cpu_audio.py tts --tts unix:PATH --sentences FILE --out DIR [--concurrency 1,2,4] [--rounds 5]
    python cpu_audio.py stt --model DIR --corpus DIR --out DIR [--concurrency 1,2,4] [--threads 8]

Each writes DIR/tts.jsonl or DIR/stt.jsonl, one line a request, and prints the summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VOICE, MODEL = "af_heart", "kokoro"  # flatshare_voice.plugins.TTS_VOICE, TTS_MODEL


def _p(values: list[float], q: int) -> float:
    xs = sorted(values)
    return xs[0] if len(xs) == 1 else statistics.quantiles(xs, n=100, method="inclusive")[q - 1]


async def _speak(client, text: str) -> dict:
    t0 = time.monotonic()
    first = None
    size = 0
    body = {"model": MODEL, "input": text, "voice": VOICE, "response_format": "pcm", "stream": True}
    async with client.stream("POST", "/v1/audio/speech", json=body, timeout=120) as r:
        r.raise_for_status()
        async for chunk in r.aiter_bytes():
            if chunk and first is None:
                first = time.monotonic()
            size += len(chunk)
    end = time.monotonic()
    audio_s = size / 2 / 24_000
    return {
        "text": text,
        "first_audio_ms": round((first - t0) * 1000, 1),
        "total_ms": round((end - t0) * 1000, 1),
        "audio_s": round(audio_s, 3),
    }


async def tts(a: argparse.Namespace) -> list[dict]:
    import httpx

    transport = httpx.AsyncHTTPTransport(uds=a.tts[5:]) if a.tts.startswith("unix:") else None
    base = "http://tts" if transport else a.tts
    sentences = [s for s in Path(a.sentences).read_text().splitlines() if s.strip()]
    rows = []
    async with httpx.AsyncClient(transport=transport, base_url=base) as client:
        await _speak(client, sentences[0])  # the first request loads what it needs
        for c in a.concurrency:
            for r in range(a.rounds):
                batch = [sentences[(r * c + i) % len(sentences)] for i in range(c)]
                for row in await asyncio.gather(*(_speak(client, s) for s in batch)):
                    rows.append({"concurrency": c, "round": r, **row})
    return rows


WORD = re.compile(r"[a-z0-9']+")


def _wer(ref: str, hyp: str) -> tuple[int, int]:
    """Word errors (edit distance over lowercase words) and the reference's words."""
    r, h = WORD.findall(ref.lower()), WORD.findall(hyp.lower())
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)], len(r)


def _pcm(path: Path):
    import numpy as np

    with wave.open(str(path)) as w:
        if w.getframerate() != 16_000 or w.getnchannels() != 1:
            raise ValueError(f"{path}: not 16 kHz mono")
        return (
            np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        )


def stt(a: argparse.Namespace) -> list[dict]:
    from faster_whisper import WhisperModel

    manifest = json.loads((Path(a.corpus) / "manifest.json").read_text())
    clips = [
        (name, e["text"], _pcm(Path(a.corpus) / f"{name}.wav"))
        for name, e in sorted(manifest.items())
        if e["kind"] == "turn"
    ]
    rows = []
    for c in a.concurrency:
        model = WhisperModel(
            a.model, device="cpu", compute_type="int8", cpu_threads=a.threads, num_workers=c
        )
        list(
            model.transcribe(clips[0][2], language="en", beam_size=1, without_timestamps=True)[0]
        )  # warm
        lock = threading.Lock()

        def one(clip, c=c, model=model, lock=lock):
            name, text, pcm = clip
            t0 = time.monotonic()
            segments, _ = model.transcribe(
                pcm, language="en", beam_size=1, without_timestamps=True, vad_filter=False
            )
            hyp = " ".join(s.text.strip() for s in segments)
            ms = (time.monotonic() - t0) * 1000
            errors, words = _wer(text, hyp)
            with lock:
                rows.append(
                    {
                        "concurrency": c,
                        "clip": name,
                        "audio_s": round(len(pcm) / 16_000, 3),
                        "transcript_ms": round(ms, 1),
                        "errors": errors,
                        "words": words,
                        "text": hyp,
                    }
                )

        with ThreadPoolExecutor(c) as pool:
            list(pool.map(one, clips[: a.clips]))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="what", required=True)
    t = sub.add_parser("tts")
    t.add_argument("--tts", required=True)
    t.add_argument("--sentences", required=True)
    t.add_argument("--rounds", type=int, default=5)
    s = sub.add_parser("stt")
    s.add_argument("--model", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--clips", type=int, default=40)
    for x in (t, s):
        x.add_argument("--out", type=Path, required=True)
        x.add_argument(
            "--concurrency", type=lambda v: [int(n) for n in v.split(",")], default=[1, 2, 4]
        )
    a = p.parse_args()
    rows = asyncio.run(tts(a)) if a.what == "tts" else stt(a)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / f"{a.what}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for c in a.concurrency:
        sel = [r for r in rows if r["concurrency"] == c]
        if a.what == "tts":
            first = [r["first_audio_ms"] for r in sel]
            rtf = [r["total_ms"] / 1000 / r["audio_s"] for r in sel if r["audio_s"]]
            print(
                f"tts c={c}: {len(sel)} requests, first audio p50 {_p(first, 50):.0f} ms p95 {_p(first, 95):.0f} ms, "
                f"real-time factor p50 {_p(rtf, 50):.2f}"
            )
        else:
            ms = [r["transcript_ms"] for r in sel]
            errors, words = sum(r["errors"] for r in sel), sum(r["words"] for r in sel)
            print(
                f"stt c={c} threads={a.threads}: {len(sel)} clips, transcript p50 {_p(ms, 50):.0f} ms p95 "
                f"{_p(ms, 95):.0f} ms, WER {100 * errors / words:.1f}%"
            )


if __name__ == "__main__":
    main()
