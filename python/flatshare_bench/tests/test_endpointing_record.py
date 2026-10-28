"""The endpointing sweep's record (benchmarks/results/2026-10-28-endpointing) against its README's table."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-28-endpointing"
SUMMARY = REPO / "benchmarks" / "endpointing" / "summary.py"


def _summary() -> ModuleType:
    spec = importlib.util.spec_from_file_location("endpointing_summary", SUMMARY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_readmes_table_is_what_the_records_give() -> None:
    readme = (RECORD / "README.md").read_text()
    found = re.search(r"<!-- summary.py -->\n(.*?)\n<!-- /summary.py -->", readme, re.S)
    assert found is not None
    assert found.group(1) == _summary().table(RECORD)


def test_each_setting_ran_the_values_its_directory_names() -> None:
    settings = sorted(RECORD.glob("silence-*-delay-*/setting.json"))
    assert settings
    for path in settings:
        setting = json.loads(path.read_text())
        name = f"silence-{setting['vad_min_silence']:g}-delay-{setting['endpointing_min_delay']:g}"
        assert path.parent.name == name
