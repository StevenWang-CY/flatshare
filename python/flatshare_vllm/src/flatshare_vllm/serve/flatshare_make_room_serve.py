"""make-room as the box serves it: the make-room module's AsyncScheduler, unchanged, that says what the engine runs.

The serving profile's variant, ``make-room-serve`` (deploy/phastform/variants), runs the engine with
``--scheduler-cls flatshare_make_room_serve.AsyncScheduler`` and this directory on PYTHONPATH beside the make-room
module that build.py writes. The class adds nothing to the scheduling: its constructor runs make-room's, then, where
FLATSHARE_ENGINE_EFFECTIVE names a file, writes the configuration the scheduler was built with to it, in the engine
core's process: the scheduler's switches as it parsed them and its policy, step cap and pool, the model, its revisions
and context, the KV connector, the vLLM version, and the SHA-256 of the scheduler module it extends and of this file.
deploy/bin/serving-check compares that with the serving profile's serving.json, because a production box serves no
/server_info (vLLM's development mode, the measurement drop-in's): the engine's own account of what it runs, read
from inside it.

The file is written whole or not at all (a temporary file renamed over it), and a write that fails fails the engine's
start, which the check would fail anyway. Without FLATSHARE_ENGINE_EFFECTIVE nothing is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import flatshare_make_room_scheduler as make_room
import vllm

SCHEMA = 1
"""The effective configuration's format; deploy/bin/serving-check refuses another."""
ENV = "FLATSHARE_ENGINE_EFFECTIVE"


def sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value: Any) -> Any:
    """``value`` as JSON holds it: enums by their value, other objects (a torch dtype) by their text."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return _plain(value.value) if hasattr(value, "value") else str(value)


def effective(scheduler: Any) -> dict[str, Any]:
    """What ``scheduler``, a make-room scheduler just built, runs with."""
    config = scheduler.vllm_config
    model, sched, cache = config.model_config, config.scheduler_config, config.cache_config
    switches = scheduler._make_room_switches(config.additional_config.get("make_room", {}))
    transfer = config.kv_transfer_config
    return {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "written_at": time.time(),
        "vllm_version": vllm.__version__,
        "scheduler_cls": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        "scheduler_module": {"name": make_room.__name__, "sha256": sha256(make_room.__file__)},
        "serve_module_sha256": sha256(__file__),
        "model": {
            "model": model.model,
            "revision": model.revision,
            "tokenizer": model.tokenizer,
            "tokenizer_revision": model.tokenizer_revision,
            "max_model_len": model.max_model_len,
            "dtype": _plain(model.dtype),
            "quantization": _plain(model.quantization),
        },
        "scheduler": {
            "policy": _plain(scheduler.policy),
            "max_num_batched_tokens": sched.max_num_batched_tokens,
            "max_num_scheduled_tokens": scheduler.max_num_scheduled_tokens,
            "long_prefill_token_threshold": sched.long_prefill_token_threshold,
            "max_num_seqs": sched.max_num_seqs,
            "async_scheduling": sched.async_scheduling,
            "enable_chunked_prefill": sched.enable_chunked_prefill,
        },
        "cache": {
            "block_size": scheduler.block_size,
            "num_gpu_blocks": cache.num_gpu_blocks,
            "pool_tokens": cache.num_gpu_blocks * scheduler.block_size,
            "kv_cache_memory_bytes": cache.kv_cache_memory_bytes,
            "gpu_memory_utilization": cache.gpu_memory_utilization,
            "enable_prefix_caching": cache.enable_prefix_caching,
            "cache_dtype": _plain(cache.cache_dtype),
        },
        "kv_transfer": None
        if transfer is None
        else {
            "kv_connector": transfer.kv_connector,
            "kv_role": _plain(transfer.kv_role),
            "kv_connector_extra_config": _plain(transfer.kv_connector_extra_config),
        },
        # What the scheduler holds, and the reserve as the table gave it (it holds the reserve in blocks).
        "make_room": {
            "max_preemptions_per_step": scheduler.max_priority_preemptions_per_step,
            "budget_cap": scheduler.priority_prefill_budget_cap,
            "budget_cap_tokens": scheduler.priority_prefill_budget_cap_tokens,
            "reserve_tokens": switches["reserve_tokens"],
            "reserve_blocks": scheduler.reserve_blocks,
        },
        "additional_config": _plain(dict(config.additional_config)),
    }


def write(path: str, report: dict[str, Any]) -> None:
    """``report`` to ``path``, whole: a reader finds the previous file or this one."""
    target = Path(path)
    staged = target.with_name(f".{target.name}.{os.getpid()}")
    try:
        staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        staged.replace(target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


class AsyncScheduler(make_room.AsyncScheduler):
    """make-room's AsyncScheduler, which writes the configuration it runs with once it is built."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if path := os.environ.get(ENV):
            write(path, effective(self))
