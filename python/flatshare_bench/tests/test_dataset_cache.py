"""AIPerf's dataset cache keeps ``weka_trace_scaled`` datasets, in a directory named by the plugin.

Each build runs AIPerf 0.13.0's own dataset manager offline, as ``aiperf profile`` builds an E2 point's
dataset: ``configs/agentic_replay.yaml`` as AIPerf's resolver chain resolves it, on four of the
committed 32x traces, with a tokenizer that AIPerf loads from a directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from aiperf.common import random_generator as rng
from aiperf.common.environment import Environment
from aiperf.config import BenchmarkRun
from aiperf.config.loader import build_benchmark_plan, load_config
from aiperf.config.resolution.resolvers import build_default_resolver_chain
from aiperf.dataset import mmap_cache
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.orchestrator.orchestrator import resolve_run_seed

from flatshare_bench import aiperf_plugin
from flatshare_bench.aiperf_plugin import PACKAGE, SCALED_TRACE, cache_datasets, cache_namespace

REPO = Path(__file__).resolve().parents[3]
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TRACES = REPO / "crates" / "flatshare-sim" / "testdata" / "weka-x32-sub16"
STOCK = mmap_cache.TRACE_VERBATIM_DATASET_TYPES - {SCALED_TRACE}
"""The types AIPerf 0.13.0 caches by itself."""

RESOLVE = """
import json, sys
from aiperf.config import BenchmarkRun
from aiperf.config.loader import build_benchmark_plan, load_config
from aiperf.config.resolution.resolvers import build_default_resolver_chain
from aiperf.dataset import mmap_cache

plan = build_benchmark_plan(load_config(sys.argv[1]))
cfg = plan.configs[0]
run = BenchmarkRun(benchmark_id="resolve", cfg=cfg, artifact_dir=cfg.artifacts.dir)
build_default_resolver_chain().resolve_all(run)
print(json.dumps({
    "imported": "flatshare_bench.aiperf_plugin" in sys.modules,
    "cached": mmap_cache.is_trace_or_verbatim_dataset("weka_trace_scaled", None),
    "cache_dir": str(mmap_cache.cache_dir()),
}))
"""
"""What ``aiperf profile`` does in its own process before it forks its services."""


@dataclass(frozen=True)
class Built:
    """A point's dataset as the replay reads it, and how the dataset manager came by it."""

    hit: bool
    key: str | None
    metadata: str
    """The DatasetMetadata it published, from which AIPerf schedules the replay."""
    files: tuple[bytes, bytes]
    """dataset.dat and index.dat, the memory map the workers read."""
    conversations: list[str]
    """Every conversation as a worker decodes it from the map."""
    inputs_json: bool


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> Path:
    """A word-level tokenizer of the README's words, saved as Hugging Face saves one: AIPerf loads it
    as it loads a model's, and it tokenizes the prompt corpora AIPerf builds for every dataset (5 MB
    of Shakespeare among them) in about a second."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    words = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    words.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]"], show_progress=False)
    words.train_from_iterator([(REPO / "README.md").read_text()], trainer=trainer)
    path = tmp_path_factory.mktemp("tokenizer")
    PreTrainedTokenizerFast(tokenizer_object=words, unk_token="[UNK]").save_pretrained(path)
    return path


@pytest.fixture
def point(tmp_path, monkeypatch, tokenizer) -> Path:
    """A point that replays the four smallest traces of the subset; returns the cache's base."""
    corpus = tmp_path / "weka-x32-sub4"
    corpus.mkdir()
    for trace in sorted(TRACES.glob("*.json"), key=lambda path: path.stat().st_size)[:4]:
        shutil.copy(trace, corpus)
    monkeypatch.setenv("FLATSHARE_BENCH_DATA", str(corpus))
    monkeypatch.setenv("FLATSHARE_BENCH_TOKENIZER", str(tokenizer))
    monkeypatch.setenv("FLATSHARE_BENCH_ARTIFACTS", str(tmp_path / "aiperf"))
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path / "mmap")
    monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)
    return tmp_path / "dataset_mmap"


def package(tmp_path: Path, changed: str | None = None) -> Path:
    """A copy of the files the namespace hashes, with one more line in ``changed``."""
    copy = tmp_path / f"package-{changed or 'as-is'}"
    copy.mkdir()
    for name in ("aiperf_plugin.py", "plugins.yaml"):
        shutil.copy(PACKAGE / name, copy)
    if changed:
        with (copy / changed).open("a") as f:
            f.write("\n")
    return copy


def build(monkeypatch, *, cache: Path | None) -> Built:
    """Configure one point's dataset, with AIPerf's cache in ``cache`` (None: the cache off)."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", cache is not None)
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_DIR", cache)
    plan = build_benchmark_plan(load_config(CONFIGS / "agentic_replay.yaml"))
    cfg = plan.configs[0]
    seed = resolve_run_seed(plan, plan.variations[0])
    run = BenchmarkRun(
        benchmark_id=uuid4().hex[:12], cfg=cfg, artifact_dir=cfg.artifacts.dir, random_seed=seed
    )
    build_default_resolver_chain().resolve_all(run)
    manager = DatasetManager(run=run, service_id="dataset_manager")
    published = []

    async def publish(message) -> None:
        published.append(message)

    async def configure() -> Built:
        await manager._do_profile_configure(None)
        try:
            metadata = published[-1].metadata
            data, index = manager._run_mmap_paths()
            client = manager._dataset_client
            conversations = [
                (await client.get_conversation(c.conversation_id)).model_dump_json()
                for c in metadata.conversations
            ]
            return Built(
                hit=manager._cache_hit_used,
                key=manager._cache_key_for_run,
                metadata=metadata.model_dump_json(),
                files=(data.read_bytes(), index.read_bytes()),
                conversations=conversations,
                inputs_json=(cfg.artifacts.dir / "inputs.json").exists(),
            )
        finally:
            await manager._cleanup()

    manager.publish = publish
    rng.reset()  # as the service's bootstrap seeds it
    rng.init(run.random_seed)
    try:
        return asyncio.run(configure())
    finally:
        rng.reset()


def test_the_plugin_makes_its_datasets_cacheable():
    assert mmap_cache.is_trace_or_verbatim_dataset(SCALED_TRACE, None)
    assert "weka_trace" in STOCK
    assert re.fullmatch(r"flatshare-[0-9a-f]{12}", cache_namespace())
    assert mmap_cache.cache_dir().name == cache_namespace()


def test_the_namespace_goes_under_the_configured_cache_once(tmp_path, monkeypatch):
    monkeypatch.setattr(mmap_cache, "TRACE_VERBATIM_DATASET_TYPES", STOCK)
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_DIR", tmp_path)
    cache_datasets()
    cache_datasets()
    cached = mmap_cache.TRACE_VERBATIM_DATASET_TYPES
    assert cached == STOCK | {SCALED_TRACE}
    assert mmap_cache.cache_dir() == tmp_path / cache_namespace()


def test_the_namespace_follows_the_plugin_and_the_version(tmp_path, monkeypatch):
    namespace = cache_namespace()
    assert cache_namespace(package(tmp_path)) == namespace
    assert cache_namespace(package(tmp_path, "aiperf_plugin.py")) != namespace
    assert cache_namespace(package(tmp_path, "plugins.yaml")) != namespace
    monkeypatch.setattr(aiperf_plugin.metadata, "version", lambda name: "1.0.1")
    assert cache_namespace() != namespace


def test_a_hit_replays_what_a_miss_and_an_uncached_build_replay(point, monkeypatch):
    cache = point / cache_namespace()
    miss = build(monkeypatch, cache=cache)
    hit = build(monkeypatch, cache=cache)
    # Uncached as before the plugin made its datasets cacheable, when configs/aiperf.env turned the
    # cache off.
    monkeypatch.setattr(mmap_cache, "TRACE_VERBATIM_DATASET_TYPES", STOCK)
    uncached = build(monkeypatch, cache=None)

    assert (miss.hit, hit.hit, uncached.hit) == (False, True, False)
    assert miss.key is not None and hit.key == miss.key and uncached.key is None
    assert [entry.name for entry in cache.iterdir() if entry.is_dir()] == [miss.key]
    assert len(miss.conversations) > 4  # the traces' subagents are conversations of their own
    for other in (hit, uncached):
        assert other.metadata == miss.metadata
        assert other.files == miss.files
        assert other.conversations == miss.conversations
    # AIPerf writes no inputs.json for the types it caches, as for its own weka_trace.
    assert (miss.inputs_json, hit.inputs_json, uncached.inputs_json) == (False, False, True)


def test_a_changed_plugin_does_not_reuse_a_dataset(point, monkeypatch, tmp_path):
    changed = cache_namespace(package(tmp_path, "aiperf_plugin.py"))
    before = build(monkeypatch, cache=point / cache_namespace())
    after = build(monkeypatch, cache=point / changed)
    again = build(monkeypatch, cache=point / cache_namespace())

    assert after.key == before.key  # AIPerf's own key does not see the change
    assert (before.hit, after.hit, again.hit) == (False, False, True)


def test_aiperf_profile_imports_the_plugin_before_it_forks_its_services(tmp_path):
    # The dataset manager is a service forked from the process that resolved the configuration: it
    # caches the plugin's datasets, in the plugin's directory, only if resolving imported the plugin.
    env = os.environ | {
        "FLATSHARE_BENCH_DATA": str(TRACES),
        "AIPERF_DATASET_MMAP_CACHE_DIR": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, "-c", RESOLVE, str(CONFIGS / "agentic_replay.yaml")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "imported": True,
        "cached": True,
        "cache_dir": str(tmp_path / cache_namespace()),
    }
