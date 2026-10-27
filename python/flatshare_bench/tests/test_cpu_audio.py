"""The CPU speech record (benchmarks/results/2026-10-27-cpu-audio) against its README's table, and the measure's word
error count."""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-27-cpu-audio"
SCRIPT = REPO / "benchmarks" / "audio" / "cpu_audio.py"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (RECORD / name).read_text().splitlines()]


def _p(values: list[float], q: int) -> float:
    return statistics.quantiles(sorted(values), n=100, method="inclusive")[q - 1]


def test_the_word_errors_are_an_edit_distance_over_words() -> None:
    spec = importlib.util.spec_from_file_location("cpu_audio", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    asked, heard = "Hi, do you have a table for four?", "hi do you have a table for 4"
    assert module._wer(asked, heard) == (1, 8)
    assert module._wer("one two three", "one three") == (1, 3)
    assert module._wer("one two", "one two two") == (1, 2)


def test_the_records_table() -> None:
    tts = _rows("tts.jsonl")
    found = {}
    for c in (1, 2, 4):
        s = [r for r in tts if r["concurrency"] == c]
        first = [r["first_audio_ms"] for r in s]
        rtf = statistics.median(r["total_ms"] / 1000 / r["audio_s"] for r in s)
        found[c] = [round(_p(first, 50)), round(_p(first, 95)), round(rtf, 2)]
        for name in ("stt-t8.jsonl", "stt-t4.jsonl"):
            ms = [r["transcript_ms"] for r in _rows(name) if r["concurrency"] == c]
            found[c] += [round(_p(ms, 50)), round(_p(ms, 95))]
    assert found == {
        1: [991, 1347, 0.42, 9045, 9220, 16347, 16461],
        2: [2736, 4420, 0.75, 18299, 20873, 18259, 18722],
        4: [3988, 7353, 1.28, 30505, 33655, 31100, 34411],
    }
    for name in ("stt-t8.jsonl", "stt-t4.jsonl"):
        rows = _rows(name)
        assert len(rows) == 60
        wer = sum(r["errors"] for r in rows) / sum(r["words"] for r in rows)
        assert f"{wer:.1%}" == "3.8%"
        assert round(statistics.median(r["audio_s"] for r in rows), 2) == 2.76
