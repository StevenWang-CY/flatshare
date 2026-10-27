"""The first-sentence measure (benchmarks/replies/first_sentence.py) against what it stands for, and its record: the
script's prompt is the voice worker's instructions and its sentence rule the bench's (LiveKit Agents' default), and the
README's table is what benchmarks/results/2026-10-27-first-sentence/replies.jsonl gives."""

from __future__ import annotations

import ast
import importlib.util
import json
import statistics
import sys
import types
from pathlib import Path

from flatshare_bench import voice_turn

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "benchmarks" / "replies" / "first_sentence.py"
RECORD = REPO / "benchmarks" / "results" / "2026-10-27-first-sentence"


def _script() -> types.ModuleType:
    # The script needs httpx only to ask an engine; its counting does not.
    sys.modules.setdefault("httpx", types.ModuleType("httpx"))
    spec = importlib.util.spec_from_file_location("first_sentence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_instructions() -> str:
    worker = REPO / "python" / "flatshare_voice" / "src" / "flatshare_voice" / "worker.py"
    for node in ast.parse(worker.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "INSTRUCTIONS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("no INSTRUCTIONS in worker.py")


def test_the_script_asks_as_the_worker_and_splits_as_livekit() -> None:
    script = _script()
    assert _worker_instructions() == script.INSTRUCTIONS
    assert script.SENTENCE_END.pattern == voice_turn.SENTENCE_END.pattern
    assert script.MIN_SENTENCE_CHARS == voice_turn.MIN_SENTENCE_CHARS
    assert len(script.turn_lines()) == 40
    # The sentence is complete once the character after its end arrives, or at the reply's end.
    tokens = [b"Yes", b",", b" we", b" have", b" a", b" table", b" at", b" seven", b".", b" Would"]
    assert script.first_sentence_tokens(tokens) == (10, True)
    assert script.first_sentence_tokens(tokens[:9]) == (9, True)
    # Shorter than 20 characters to the end: spoken whole.
    assert script.first_sentence_tokens([b"Sure", b".", b" Ok"]) == (3, False)
    for text in ("Yes, we have a table at seven. Would you", "Sure. Ok"):
        expected = voice_turn.first_sentence(text)
        end = script.first_sentence_end(text)
        assert (text[:end].strip() if end is not None else None) == expected


def test_the_records_table() -> None:
    rows = [json.loads(line) for line in (RECORD / "replies.jsonl").read_text().splitlines()]
    assert len(rows) == 320
    found = {}
    for kind in ("single", "conversation", None):
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        first = sorted(r["first_sentence_tokens"] for r in sel)
        whole = sorted(r["completion_tokens"] for r in sel)
        p95 = statistics.quantiles(first, n=100, method="inclusive")[94]
        found[kind] = (
            len(sel),
            statistics.median(first),
            round(p95),
            first[-1],
            sum(x <= 16 for x in first),
            sum(x <= 48 for x in first),
            statistics.median(whole),
            whole[-1],
        )
    assert found == {
        "single": (120, 14, 26, 31, 79, 120, 26, 48),
        "conversation": (200, 16, 30, 39, 101, 200, 26, 43),
        None: (320, 15, 29, 39, 180, 320, 26, 48),
    }
    assert all(r["finish_reason"] == "stop" for r in rows)
