"""How many tokens the voice agent's replies take to finish their first sentence, which make-room's first-tokens window
has to cover so that speech synthesis starts while lower-priority prefill is held.

Asks the engine for the voice agent's replies to the caller corpus's lines, with the worker's instructions
(flatshare_voice.worker.INSTRUCTIONS) and the engine's default sampling, and reads each reply's tokens back
(``logprobs``): the first sentence is LiveKit Agents' (flatshare_bench.voice_turn.first_sentence: up to the first
sentence end that leaves at least 20 characters), and its tokens are the reply's tokens up to the one that completes
it. A reply shorter than that is spoken whole, so its first sentence is all of its tokens. Two kinds of turn: each
corpus line on its own, and conversations of 25 turns that keep the history, as a call's chat context does.

  python first_sentence.py --engine unix:/run/user/UID/flatshare/engine.sock --out DIR
Run with the engine's environment (httpx). Writes DIR/replies.jsonl and prints the summary.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
CALLERS = HERE.parents[1] / "python" / "flatshare_voice" / "src" / "flatshare_voice" / "callers.py"


def turn_lines(path: Path = CALLERS) -> tuple[str, ...]:
    """The caller corpus's TURN_LINES, read from its source: importing it would need the voice worker's environment."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "TURN_LINES" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"no TURN_LINES in {path}")


# flatshare_voice.worker.INSTRUCTIONS, which imports LiveKit; kept equal by tests/test_first_sentence.py.
INSTRUCTIONS = (
    "You are Ava, the phone assistant of a small neighborhood restaurant. You take reservations, "
    "answer questions about the menu, opening hours and directions, and keep every answer to one "
    "or two short, natural sentences. You are speaking, so never use lists, markdown or emoji."
)
# flatshare_bench.voice_turn's sentence rule, LiveKit Agents' tokenizer default; kept equal by the same test.
SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?:\s|$)")
MIN_SENTENCE_CHARS = 20


def first_sentence_end(text: str) -> int | None:
    """The end of the first sentence in ``text`` that leaves at least MIN_SENTENCE_CHARS to speak, or None."""
    for m in SENTENCE_END.finditer(text):
        if len(text[: m.end()].strip()) >= MIN_SENTENCE_CHARS:
            return m.end()
    return None


def first_sentence_tokens(tokens: list[bytes]) -> tuple[int, bool]:
    """How many of a reply's tokens it takes to complete its first sentence, and whether it has one: a sentence end
    needs the character after it (a space) or the reply's end, as a streamed reply shows it."""
    text = b""
    for k, token in enumerate(tokens, 1):
        text += token
        decoded = text.decode("utf-8", errors="ignore")
        end = first_sentence_end(decoded)
        if end is not None and (end < len(decoded) or k == len(tokens)):
            return k, True
    return len(tokens), False


def reply(client: httpx.Client, model: str, messages: list[dict]) -> dict:
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": 256,
            "logprobs": True,
            "top_logprobs": 0,
        },
        timeout=120,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    tokens = [
        bytes(t["bytes"]) if t.get("bytes") is not None else t["token"].encode()
        for t in choice["logprobs"]["content"]
    ]
    k, ended = first_sentence_tokens(tokens)
    return {
        "text": choice["message"]["content"],
        "completion_tokens": len(tokens),
        "first_sentence_tokens": k,
        "has_sentence_end": ended,
        "finish_reason": choice["finish_reason"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--engine", required=True, help="unix:PATH of the engine's socket, or http://HOST:PORT"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=3, help="replies to each line on its own")
    p.add_argument("--conversations", type=int, default=8)
    p.add_argument("--turns", type=int, default=25)
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--callers", type=Path, default=CALLERS, help="the caller corpus's callers.py")
    a = p.parse_args()
    lines = turn_lines(a.callers)
    if a.engine.startswith("unix:"):
        client = httpx.Client(
            transport=httpx.HTTPTransport(uds=a.engine[5:]), base_url="http://engine"
        )
    else:
        client = httpx.Client(base_url=a.engine)
    model = client.get("/v1/models").json()["data"][0]["id"]
    rows = []
    for i, line in enumerate(lines):
        for s in range(a.samples):
            msgs = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": line}]
            rows.append({"kind": "single", "line": i, "sample": s, **reply(client, model, msgs)})
    rng = random.Random(a.seed)
    for c in range(a.conversations):
        msgs = [{"role": "system", "content": INSTRUCTIONS}]
        for t in range(a.turns):
            i = rng.randrange(len(lines))
            msgs.append({"role": "user", "content": lines[i]})
            row = reply(client, model, msgs)
            msgs.append({"role": "assistant", "content": row["text"]})
            rows.append({"kind": "conversation", "conversation": c, "turn": t, "line": i, **row})
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "replies.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for kind in ("single", "conversation", "all"):
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        fs = sorted(r["first_sentence_tokens"] for r in sel)
        whole = sorted(r["completion_tokens"] for r in sel)
        q = statistics.quantiles(fs, n=100, method="inclusive")
        print(
            f"{kind}: {len(sel)} replies, model {model}; first sentence tokens median {statistics.median(fs)}, "
            f"p95 {q[94]:.1f}, max {fs[-1]}; within 16: {sum(x <= 16 for x in fs)}, within 48: "
            f"{sum(x <= 48 for x in fs)}; whole reply median {statistics.median(whole)}, max {whole[-1]}; "
            f"without a sentence end {sum(not r['has_sentence_end'] for r in sel)}"
        )


if __name__ == "__main__":
    main()
