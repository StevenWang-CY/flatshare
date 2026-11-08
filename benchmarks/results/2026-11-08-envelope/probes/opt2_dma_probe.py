"""opt2_dma_probe.py: whether a CPU-tier load delays a concurrent engine step on the same card (hypothesis B), on the bare
card with the engine's own packages. Foreground, every 15 ms on the default stream, a step: a 64 KiB pinned H2D copy (the
step's inputs), a memory-bound kernel of about 5 ms (a decode's weight reads), a 64-byte D2H copy (the sampled tokens),
timed from enqueue to completion. Background, on a second stream, one load as vLLM 0.32.0 issues it: _C_cache_ops
.swap_blocks_batch from pinned host pages of 32 KiB to scattered device pages (cuMemcpyBatchAsync, source order any), or the
Triton kernel on 16 KiB pages (fp8 KV's path, 12 SMs). Prints the steps' latency alone and over each load, and the load's
time and rate."""
import sys
import time

import numpy as np
import torch
import vllm._custom_ops  # noqa: F401  registers _C_cache_ops
from vllm.v1.kv_offload.cpu.swap_blocks_triton import swap_blocks_batch as triton_swap

PAGE = 32 * 1024
HOST_BYTES = int(3.5e9)
dev = torch.device("cuda")
host = torch.empty(HOST_BYTES, dtype=torch.uint8, pin_memory=True)
host.fill_(1)
device_buf = torch.empty(HOST_BYTES, dtype=torch.uint8, device=dev)
weights = torch.empty(int(1.2e9), dtype=torch.bfloat16, device=dev)
inp_host = torch.ones(16 * 1024, dtype=torch.int32, pin_memory=True)
inp_dev = torch.empty_like(inp_host, device=dev)
out_dev = torch.zeros(16, dtype=torch.int32, device=dev)
out_host = torch.empty(16, dtype=torch.int32, pin_memory=True)
side = torch.cuda.Stream()
rng = np.random.default_rng(0)


def step():
    start = time.perf_counter()
    inp_dev.copy_(inp_host, non_blocking=True)
    s = weights.sum()
    out_dev[0] = s.to(torch.int32) + inp_dev[0]
    out_host.copy_(out_dev, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    event.synchronize()
    return (time.perf_counter() - start) * 1e3


def descriptors(nbytes, page):
    n = nbytes // page
    src = torch.from_numpy(host.data_ptr() + np.arange(n, dtype=np.int64) * page)
    slots = rng.permutation(HOST_BYTES // page)[:n].astype(np.int64)
    dst = torch.from_numpy(device_buf.data_ptr() + slots * page)
    sizes = torch.full((n,), page, dtype=torch.int64)
    return src, dst, sizes


def run(label, nbytes=0, page=PAGE, parts=1, triton=False):
    loads = []
    if nbytes:
        src, dst, sizes = descriptors(nbytes, page)
        cut = np.linspace(0, len(src), parts + 1).astype(int)
        loads = [(src[a:b], dst[a:b], sizes[a:b]) for a, b in zip(cut[:-1], cut[1:])]
    lat_before, lat_during, lat_after = [], [], []
    begin = end = None
    for i in range(10):
        lat_before.append(step())
        time.sleep(0.015)
    if loads:
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            begin.record(side)
            for s, d, z in loads:
                if triton:
                    triton_swap(s, d, z, bytes_per_chunk=page)
                else:
                    torch.ops._C_cache_ops.swap_blocks_batch(s, d, z, True)
            end.record(side)
        while not end.query():
            lat_during.append(step())
            time.sleep(0.015)
    else:
        for i in range(40):
            lat_during.append(step())
            time.sleep(0.015)
    for i in range(10):
        lat_after.append(step())
        time.sleep(0.015)
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")
    load = ""
    if loads:
        ms = begin.elapsed_time(end)
        load = f"; load {nbytes / 1e9:.2f} GB in {parts} batch(es) of {page // 1024} KiB pages: {ms:.0f} ms, {nbytes / ms / 1e6:.1f} GB/s"
    print(f"{label}: steps before p50 {q(lat_before, .5):.1f} ms; during n={len(lat_during)} p50/p95/max {q(lat_during, .5):.1f}/{q(lat_during, .95):.1f}/{max(lat_during):.1f} ms; after p50 {q(lat_after, .5):.1f}{load}", flush=True)


torch.cuda.synchronize()
for _ in range(20):
    step()
run("alone")
run("load 1.46 GB, one batch", int(1.46e9))
run("load 3.4 GB, one batch", int(3.4e9))
run("load 1.46 GB, 23 batches of 64 MB", int(1.46e9), parts=23)
run("load 1.46 GB, triton 16 KiB pages", int(1.46e9), page=16 * 1024, triton=True)
run("alone again")
print(torch.cuda.get_device_name(), "max memory", torch.cuda.max_memory_allocated() / 2**30, "GiB")
