"""opt2_dma_probe3.py: probe 2 showed a step's 64 KiB H2D copy waits behind a load's DMA to its end (208 ms for 1.46 GB at
PCIe 3.0 x8) even with 2 ms of host submission. This one asks what avoids it: (a) the load as Triton's SM copy kernel
(vLLM's fast path for pages under 28 KiB) on 32 KiB pages, warmed first (probe 1's number included the JIT), with 12 and 4
programs; (b) the DMA load cut in 64 MB jobs, each on its own stream chained by events, as the connector chains jobs."""
import threading
import time

import numpy as np
import torch
import vllm._custom_ops  # noqa: F401
import vllm.v1.kv_offload.cpu.swap_blocks_triton as st

HOST_BYTES = int(1.6e9)
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


def descriptors(nbytes, page):
    n = nbytes // page
    src = torch.from_numpy(host.data_ptr() + np.arange(n, dtype=np.int64) * page)
    slots = rng.permutation(HOST_BYTES // page)[:n].astype(np.int64)
    dst = torch.from_numpy(device_buf.data_ptr() + slots * page)
    return src, dst, torch.full((n,), page, dtype=torch.int64)


def measure(label, submit_all):
    done, stats = threading.Event(), {}

    def worker():
        stats.update(submit_all())
        done.set()

    before = [step() for _ in range(10)]
    thread = threading.Thread(target=worker)
    thread.start()
    during = []
    while not done.is_set():
        during.append(step())
        time.sleep(0.005)
    thread.join()
    print(f"   {label}: steps before p50 {q(before, .5):.1f}; during n={len(during)} p50/p95/max {q(during, .5):.1f}/{q(during, .95):.1f}/{max(during):.1f} ms; copy {stats['copy']:.0f} ms, {1.46e9 / stats['copy'] / 1e6:.1f} GB/s", flush=True)


src, dst, sizes = descriptors(int(1.46e9), 32768)
torch.cuda.synchronize()
for _ in range(20):
    step()


def triton_load(programs):
    def run():
        st.NUM_SMS = programs
        stream = torch.cuda.Stream()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            begin.record(stream)
            st.swap_blocks_batch(src, dst, sizes, bytes_per_chunk=8192)
            end.record(stream)
        end.synchronize()
        return {"copy": begin.elapsed_time(end)}
    return run


def dma_jobs(parts):
    def run():
        cut = np.linspace(0, len(src), parts + 1).astype(int)
        first, last = None, None
        for a, b in zip(cut[:-1], cut[1:]):
            stream = torch.cuda.Stream()
            if last is not None:
                stream.wait_event(last)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                begin.record(stream)
                torch.ops._C_cache_ops.swap_blocks_batch(src[a:b], dst[a:b], sizes[a:b], True)
                end.record(stream)
            first = first or begin
            last = end
        last.synchronize()
        return {"copy": first.elapsed_time(last)}
    return run


print("warm-up of the Triton kernel")
st.swap_blocks_batch(src[:4096], dst[:4096], sizes[:4096], bytes_per_chunk=8192)
torch.cuda.synchronize()
print("(a) Triton SM copy, 32 KiB pages, 1.46 GB")
measure("12 programs", triton_load(12))
measure("4 programs", triton_load(4))
measure("24 programs", triton_load(24))
print("(b) DMA in jobs on chained streams, 1.46 GB")
measure("1 job", dma_jobs(1))
measure("23 jobs of 64 MB", dma_jobs(23))
measure("92 jobs of 16 MB", dma_jobs(92))
print("alone again:", f"{q([step() for _ in range(30)], .5):.1f} ms p50")
