"""The endpointing sweep (benchmarks/endpointing): its summary's table from a sweep's records, made by hand here."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
SUMMARY = REPO / "benchmarks" / "endpointing" / "summary.py"


def _summary() -> ModuleType:
    spec = importlib.util.spec_from_file_location("endpointing_summary", SUMMARY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setting(
    sweep: Path, silence: float, delay: float, v2v: list[float | None], cut: set[int]
) -> None:
    here = sweep / f"silence-{silence}-delay-{delay}"
    (here / "bots").mkdir(parents=True)
    (here / "setting.json").write_text(
        json.dumps({"vad_min_silence": silence, "endpointing_min_delay": delay})
    )
    turns = [{"turn": 0, "clip": "turn-09", "v2v_ms": 9999.0, "warmup": True}]
    for n, ms in enumerate(v2v, 1):
        turn = {"turn": n, "clip": f"turn-{n:02}", "spoke_during_turn": n in cut}
        turn |= {"v2v_ms": ms} if ms is not None else {"error": "no reply"}
        turns.append(turn)
    (here / "bots" / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    worker = [
        "registered worker",
        json.dumps({"name": "flatshare_voice.turns", "turn": 0, "end_of_turn_ms": 1.0}),
    ]
    worker += [
        json.dumps(
            {
                "name": "flatshare_voice.turns",
                "turn": n,
                "end_of_turn_ms": 1000 * (silence + delay) + n,
            }
        )
        for n in range(1, len(v2v) + 1)
    ]
    (here / "voice-worker.log").write_text("\n".join(worker))


def test_the_sweep_table_gives_each_settings_latency_endpointing_and_cut_lines(
    tmp_path: Path,
) -> None:
    _setting(tmp_path, 0.3, 0.2, [1100.0, 1200.0, 1300.0, 1400.0], cut={2, 4})
    _setting(tmp_path, 0.55, 0.5, [1600.0, 1700.0, None, 1900.0], cut=set())
    table = _summary().table(tmp_path).splitlines()
    # The longest silence and delay first; the warm-up turn and its turn line are left out; a reply that never
    # came stands at the reply timeout in the percentiles and counts as failed.
    assert table[2:] == [
        "| 0.55 | 0.5 | 4 (1) | 1,800 / 25,785 | 1,052 / 1,054 | 0 |",
        "| 0.3 | 0.2 | 4 (0) | 1,250 / 1,385 | 502 / 504 | 2: turn-02, turn-04 |",
    ]
