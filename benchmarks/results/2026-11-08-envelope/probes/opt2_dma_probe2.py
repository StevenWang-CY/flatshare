"""opt2_dma_probe2.py: the first probe's load held the host: one step ran during a 299 ms load. This one separates the
host's submission from the copy: (1) the host time of one swap_blocks_batch call by descriptor count and page size; (2) the
load submitted from a second thread while the main thread keeps stepping, so the steps measure the copy engine and any
driver lock, not a blocked caller; (3) the same with 64 MB batches."""
import threading
import time

import numpy as np
import torch
import vllm._custom_ops  # noqa: F401

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
rng = np.random.default_rng(0)
q = lambda v, p: sorted(v)[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")


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


def descriptors(nbytes, page, contiguous=False):
    n = nbytes // page
    src = torch.from_numpy(host.data_ptr() + np.arange(n, dtype=np.int64) * page)
    slots = np.arange(n, dtype=np.int64) if contiguous else rng.permutation(HOST_BYTES // page)[:n].astype(np.int64)
    dst = torch.from_numpy(device_buf.data_ptr() + slots * page)
    return src, dst, torch.full((n,), page, dtype=torch.int64)


def submit(src, dst, sizes, stream):
    with torch.cuda.stream(stream):
        start = time.perf_counter()
        torch.ops._C_cache_ops.swap_blocks_batch(src, dst, sizes, True)
        host_ms = (time.perf_counter() - start) * 1e3
    return host_ms


side = torch.cuda.Stream()
torch.cuda.synchronize()
for _ in range(20):
    step()

print("(1) host time of one call, then the copy's own time")
for nbytes, page in ((int(64e6), 32768), (int(0.37e9), 32768), (int(1.46e9), 32768), (int(1.46e9), 65536), (int(1.46e9), 2 * 2**20)):
    src, dst, sizes = descriptors(nbytes, page)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(side)
    host_ms = submit(src, dst, sizes, side)
    end.record(side)
    end.synchronize()
    print(f"   {nbytes / 1e9:.2f} GB in {len(src)} descriptors of {page // 1024} KiB: host {host_ms:.1f} ms ({1e3 * host_ms / len(src):.2f} us each), copy {begin.elapsed_time(end):.0f} ms", flush=True)

print("(2) the load from a second thread, the main thread stepping")
for label, nbytes, page, parts in (("1.46 GB one batch, 32 KiB", int(1.46e9), 32768, 1), ("1.46 GB in 23 batches", int(1.46e9), 32768, 23), ("1.46 GB, 2 MiB pages", int(1.46e9), 2 * 2**20, 1)):
    src, dst, sizes = descriptors(nbytes, page)
    cut = np.linspace(0, len(src), parts + 1).astype(int)
    done = threading.Event()
    stats = {}

    def worker():
        stream = torch.cuda.Stream()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        host_ms = 0.0
        for a, b in zip(cut[:-1], cut[1:]):
            host_ms += submit(src[a:b], dst[a:b], sizes[a:b], stream)
        end.record(stream)
        stats["host"] = host_ms
        end.synchronize()
        stats["copy"] = begin.elapsed_time(end)
        done.set()

    before = [step() for _ in range(10)]
    thread = threading.Thread(target=worker)
    t0 = time.perf_counter()
    thread.start()
    during = []
    while not done.is_set():
        during.append(step())
        time.sleep(0.005)
    thread.join()
    wall = (time.perf_counter() - t0) * 1e3
    print(f"   {label}: steps before p50 {q(before, .5):.1f}; during n={len(during)} p50/p95/max {q(during, .5):.1f}/{q(during, .95):.1f}/{max(during):.1f} ms; submit host {stats['host']:.0f} ms, copy {stats['copy']:.0f} ms, wall {wall:.0f} ms", flush=True)
print("alone again:", f"{q([step() for _ in range(30)], .5):.1f} ms p50")
