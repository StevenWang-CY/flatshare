"""AIPerf 0.13.0 plugins, registered through ``plugins.yaml`` (entry point group ``aiperf.plugins``).

``flatshare_chat`` is AIPerf's OpenAI chat endpoint plus Flatshare's per-session metadata (see
:mod:`flatshare_bench.protocol`). AIPerf calls the endpoint for every request, including every
turn of a multi-turn Weka trace.

``weka_trace_scaled`` is AIPerf's Weka loader with one fix: it honours each trace file's own
``block_size``. AIPerf passes the registry default ``default_block_size: 64`` to ``weka_trace`` as an
override, so the stock loader re-expands a block-scaled trace to 64 tokens per block and synthesizes
prompts from only the first 1/32 of each request's hash ids. On a 32x-scaled corpus that reproduced
about 77% of the shared prefix and no assistant turns.

Importing the module also lets AIPerf's dataset cache keep ``weka_trace_scaled`` datasets
(:func:`cache_datasets`), so that the points of an E2 round, which replay one corpus with one seed,
build their dataset once.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from aiperf.common.environment import Environment
from aiperf.common.models import RequestInfo
from aiperf.dataset import mmap_cache
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.endpoints.openai_chat import ChatEndpoint

from flatshare_bench.protocol import SessionTagger

SCALED_TRACE: Final = "weka_trace_scaled"
"""The loader's name in ``plugins.yaml``: the dataset ``format`` that selects it."""

PACKAGE: Final = Path(__file__).parent
"""This module's directory, which also holds ``plugins.yaml``."""


def session_id(request_info: RequestInfo) -> str:
    """The session-tree root, so a parent agent and its subagents share one session id.

    AIPerf mints a fresh id per session play in concurrency and ``agentic_replay`` modes.
    """
    return request_info.root_correlation_id or request_info.x_correlation_id


class FlatshareChatEndpoint(ChatEndpoint):
    """OpenAI chat completions with ``X-Session-ID``, ``X-Flatshare-Class`` and ``priority`` per session."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tagger = SessionTagger.from_env()

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        headers = super().get_endpoint_headers(request_info)
        headers.update(self.tagger.headers(session_id(request_info)))
        return headers

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        payload = super().format_payload(request_info)
        payload.update(self.tagger.body(session_id(request_info)))
        return payload


class WekaTraceScaledLoader(WekaTraceLoader):
    """``weka_trace`` that reads each trace's own ``block_size`` instead of a forced 64."""

    def __init__(self, *, default_block_size: int | None = None, **kwargs: Any) -> None:
        # AIPerf's composer passes the registry default as an override; drop it.
        del default_block_size
        super().__init__(**kwargs)


def cache_namespace(package: Path = PACKAGE) -> str:
    """The directory under AIPerf's dataset cache that holds this plugin's datasets: ``flatshare-``
    and a hash of the package's version, this module and ``plugins.yaml``.

    AIPerf's cache key covers the corpus's bytes, the tokenizer's name and revision, the seeds, the
    dataset's settings and AIPerf's own build, but no plugin's code: a dataset built before a change to
    the loader, or to its registration (which picks the corpus its prompts are drawn from), is not
    served after it.
    """
    digest = hashlib.sha256(metadata.version("flatshare-bench").encode())
    for name in ("aiperf_plugin.py", "plugins.yaml"):
        digest.update(b"\0" + (package / name).read_bytes())
    return f"flatshare-{digest.hexdigest()[:12]}"


def cache_datasets() -> None:
    """Add ``weka_trace_scaled`` to the types AIPerf's dataset cache keeps, with its datasets in the
    directory :func:`cache_namespace` names under the cache's own.

    AIPerf 0.13.0 caches only the types in ``mmap_cache.TRACE_VERBATIM_DATASET_TYPES``, its own trace
    replays, and reads that set and its cache directory when it builds a dataset, not when it is
    imported. ``aiperf profile`` imports this module while it resolves its configuration (it asks the
    loader for its sampling), before it forks its services: its dataset manager inherits both. Under
    ``spawn`` it would not, and would build every dataset anew, as before. AIPerf writes no
    ``inputs.json`` for the types it caches, as for its own ``weka_trace``.
    """
    types = mmap_cache.TRACE_VERBATIM_DATASET_TYPES
    mmap_cache.TRACE_VERBATIM_DATASET_TYPES = types | {SCALED_TRACE}
    base = mmap_cache.cache_dir()
    if base.name != (namespace := cache_namespace()):
        Environment.DATASET.MMAP_CACHE_DIR = base / namespace


cache_datasets()
