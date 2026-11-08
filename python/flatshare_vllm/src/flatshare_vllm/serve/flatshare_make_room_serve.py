"""make-room as the box serves it: the make-room module's AsyncScheduler, with what the profile installs, that says what
the engine runs.

The serving profile's variant, ``make-room-serve`` (deploy/phastform/variants), runs the engine with
``--scheduler-cls flatshare_make_room_serve.AsyncScheduler`` and this directory on PYTHONPATH beside the make-room
module that build.py writes. Its constructor runs make-room's, installs what the engine's additional config asks for
(below), then, where FLATSHARE_ENGINE_EFFECTIVE names a file, writes the configuration the scheduler was built with to
it, in the engine core's process: the scheduler's switches as it parsed them and its policy, step cap and pool, the
model, its revisions and context, the KV connector, what it installed, the vLLM version, and the SHA-256 of the
scheduler module it extends, of this file and of the modules it installs from.
deploy/bin/serving-check compares that with the serving profile's serving.json, because a production box serves no
/server_info (vLLM's development mode, the measurement drop-in's): the engine's own account of what it runs, read
from inside it.

The file is written whole or not at all (a temporary file renamed over it), and a write that fails fails the engine's
start, which the check would fail anyway. Without FLATSHARE_ENGINE_EFFECTIVE nothing is written.

What it installs, each where the engine's additional config asks for it, and each recorded in the effective
configuration (off as the profile writes off, 0):
- ``flatshare_jit`` with a margin above 0 (the profile's jit_margin_ms and step-time model): flatshare_jit's
  just-in-time scheduling of the next batch on vLLM's engine core, before the core binds its step function.
- ``flatshare_queue.max_wait_s`` above 0 (the profile's queue_max_wait_s): flatshare_queue's waiting queue, the agents
  ordered by the prefill each has left net of what the GPU pool and the CPU tier hold, in place of first come first
  served, which over an LRU tier recomputes nearly every agent's prompt under load; schedule() fixes its order at the
  start of each step.
- ``flatshare_copy.sm_load_programs`` above 0 (the profile's sm_load_programs): flatshare_copy's CPU-tier loads through
  vLLM's SM copy kernel, so that a load does not hold a concurrent step's inputs on the copy engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import flatshare_copy
import flatshare_jit
import flatshare_make_room_scheduler as make_room
import flatshare_queue
import vllm

HELPERS = (flatshare_copy, flatshare_jit, flatshare_queue)
"""The modules this one installs from, whose SHA-256 the effective configuration gives beside this file's."""

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


def jit_settings(config: Any) -> dict[str, Any] | None:
    """The just-in-time scheduling the engine's additional config asks for, as numbers (a margin in ms and the step
    model's coefficients), or None where it asks for none or for a margin of 0."""
    jit = (config.additional_config or {}).get("flatshare_jit")
    if not jit or not jit.get("margin_ms"):
        return None
    return {
        "margin_ms": int(jit["margin_ms"]),
        "step_cost": {key: float(jit["step_cost"][key]) for key in flatshare_jit.STEP_COST},
    }


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
        "serve_modules_sha256": {Path(module.__file__).name: sha256(module.__file__) for module in HELPERS},
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
        # What the scheduler holds, and the reserve and the retention as the table gave them (it holds both in blocks).
        "make_room": {
            "max_preemptions_per_step": scheduler.max_priority_preemptions_per_step,
            "budget_cap": scheduler.priority_prefill_budget_cap,
            "budget_cap_tokens": scheduler.priority_prefill_budget_cap_tokens,
            "reserve_tokens": switches["reserve_tokens"],
            "reserve_blocks": scheduler.reserve_blocks,
            "take_reservations": scheduler.take_reservations,
            "decode_hold_tokens": scheduler.decode_hold_tokens,
            "retain_tokens": switches["retain_tokens"],
            "retain_blocks": scheduler.retain_blocks,
            "retain_s": scheduler.retain_s,
            "hint_hold_ms": switches["hint_hold_ms"],
        },
        # What the constructor installed on the engine core: off is a margin of 0, the profile's own way to say it.
        "jit": jit_settings(config) or {"margin_ms": 0, "step_cost": None},
        # The waiting queue the constructor installed (vLLM's own is a bound of 0) and the CPU tier's load path it
        # installed (vLLM's choice is 0 programs), as installed, not as asked.
        "queue": {
            "max_wait_s": scheduler.waiting.max_wait_s
            if isinstance(scheduler.waiting, flatshare_queue.CacheAwareQueue)
            else 0,
            "waiting_queue": f"{type(scheduler.waiting).__module__}.{type(scheduler.waiting).__qualname__}",
        },
        "copy": scheduler.flatshare_copy or {"sm_load_programs": 0},
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
    """make-room's AsyncScheduler with what the additional config installs, which writes the configuration it runs
    with once it is built."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if jit := jit_settings(self.vllm_config):
            from vllm.v1.engine import core

            cost = flatshare_jit.StepCost(**jit["step_cost"])
            flatshare_jit.install(core.EngineCore, jit["margin_ms"] / 1e3, cost)
        if max_wait_s := flatshare_queue.max_wait_s(self.vllm_config):
            # Before any request arrives: the queue make-room built is empty.
            self.waiting = flatshare_queue.CacheAwareQueue(self, max_wait_s)
        self.flatshare_copy = None
        if programs := flatshare_copy.programs(self.vllm_config):
            self.flatshare_copy = flatshare_copy.install(programs)
        if path := os.environ.get(ENV):
            write(path, effective(self))

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self.waiting, flatshare_queue.CacheAwareQueue):
            self.waiting.begin_step()
        return super().schedule(*args, **kwargs)
