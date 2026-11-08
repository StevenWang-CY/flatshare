"""flatshare_copy: the CPU tier's loads through vLLM's SM copy kernel instead of the copy engine.

On the box (an RTX 5060 Ti on PCIe 3.0 x8, 6.8 GB/s) vLLM 0.32.0 loads a request's prefix from the CPU tier with one
cuMemcpyBatchAsync over its pages, and a step that runs meanwhile waits for that copy to end before its own inputs reach
the card: 192-209 ms beside a 1.46 GB load, whatever the page size or the size of the jobs it is cut into. Beside
flatshare_queue's loads that was the calls' slow first tokens. vLLM's own Triton kernel for small pages copies through
the SMs instead: with 4 programs it moved the same 1.46 GB at 6.1 GB/s while the step beside it took 6.1 ms against 5.7
alone (benchmarks/results/2026-11-08-envelope).

The serve module (flatshare_make_room_serve) installs it at the scheduler's construction where the engine's additional
config gives ``flatshare_copy.sm_load_programs`` above 0 (the serving profile's sm_load_programs); on the uni executor
the scheduler shares the process with the worker. The load handler's copy function becomes the one vLLM's own
selection returns with its size threshold lifted (vLLM 0.32.0 sends pages of 28 KiB or more, the box's 64 KiB among
them, to the copy engine), so that its other conditions stand: pinned host memory, Triton, the platform, pages of whole
8-byte words. The kernel is compiled and checked before the first load. Stores keep the copy engine, as vLLM chooses
for them.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Any

import torch

logger = logging.getLogger("vllm.flatshare_copy")


def programs(config: Any) -> int:
    """The SM copy kernel's programs the engine's additional config asks for; 0 where it asks for vLLM's choice."""
    copy = (config.additional_config or {}).get("flatshare_copy") or {}
    count = int(copy.get("sm_load_programs", 0))
    if count < 0:
        raise ValueError(f"flatshare_copy.sm_load_programs must be 0 (off) or more, not {count}")
    return count


def load_handler() -> tuple[Any, Any]:
    """This process's CPU-tier worker and its load handler, from the worker-side connector, or Nones."""
    from vllm.distributed.kv_transfer import kv_transfer_state

    agent = kv_transfer_state._KV_CONNECTOR_AGENT
    worker = getattr(getattr(agent, "connector_worker", None), "worker", None)
    return worker, getattr(worker, "_load_handler", None)


def install(programs: int) -> dict[str, Any]:
    """Route the CPU tier's loads through ``programs`` programs of the SM copy kernel; what was installed, for the
    effective configuration."""
    from vllm.v1.kv_offload.cpu import gpu_worker, swap_blocks_triton

    worker, handler = load_handler()
    if handler is None:
        raise RuntimeError(
            "flatshare_copy: no CPU-tier load handler in this process: it needs the CPU tier and the uni executor"
        )
    region = getattr(worker, "_mmap_region", None)
    pinned = gpu_worker.PIN_MEMORY and (region is None or region.is_pinned)
    threshold = gpu_worker.THRESHOLD_BYTES
    gpu_worker.THRESHOLD_BYTES = math.inf
    try:
        copy = gpu_worker._select_swap_blocks_fn(handler.layer_refs_per_group, False, pinned)
    finally:
        gpu_worker.THRESHOLD_BYTES = threshold
    if not isinstance(copy, functools.partial):
        raise RuntimeError(
            "flatshare_copy: vLLM's selection keeps the copy engine here (host memory not pinned, no Triton, or pages "
            "not of whole words)"
        )
    swap_blocks_triton.NUM_SMS = programs
    pages = sorted({ref.page_size_bytes for group in handler.layer_refs_per_group for ref in group})
    warm(copy, pages[-1])
    handler._swap_blocks_batch = copy
    report = {
        "sm_load_programs": programs,
        "bytes_per_chunk": copy.keywords["bytes_per_chunk"],
        "page_bytes": pages,
    }
    logger.info("flatshare_copy: the CPU tier's loads copy through the SM kernel: %s", report)
    return report


WARM_JOBS = (0, 1)
"""What warm() adds to the kernel's least job count, MIN_N (16): Triton specializes the job count on its divisibility by
16, so a load of a count that is not a multiple of 16 would otherwise compile a second kernel during its step."""


def warm(copy: Any, page: int) -> None:
    """Compile the kernel before the first load, both of its specializations on enough pages to take its path, and check
    what each copied."""
    from vllm.v1.kv_offload.cpu import swap_blocks_triton

    for extra in WARM_JOBS:
        n = swap_blocks_triton.MIN_N + extra
        host = torch.randint(0, 256, (n * page,), dtype=torch.uint8).pin_memory()
        device = torch.zeros(n * page, dtype=torch.uint8, device="cuda")
        offsets = torch.arange(n, dtype=torch.int64) * page
        copy(
            host.data_ptr() + offsets,
            device.data_ptr() + offsets.flip(0),
            torch.full((n,), page, dtype=torch.int64),
            is_src_access_order_any=True,
        )
        torch.cuda.synchronize()
        if not torch.equal(device.view(n, page).flip(0).cpu(), host.view(n, page)):
            raise RuntimeError(f"flatshare_copy: the SM copy kernel's check failed on {n} pages")
