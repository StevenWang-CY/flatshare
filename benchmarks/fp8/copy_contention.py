"""Does a CPU-tier transfer hold back a small host-to-GPU copy on another stream?

FlashInfer's plan() copies its page-locked buffer to the GPU with one small cudaMemcpyAsync on the compute stream; vLLM's
CPU offload moves KV blocks on streams of its own. If the card's copy engines serve one direction at a time, the small
copy waits behind a large one, and with asynchronous scheduling the host can plan again (refilling that buffer) before the
copy has run: the race's window. This times a 64 KiB pinned copy alone and while 1 GiB moves host-to-GPU or GPU-to-host
on another stream, each time from its enqueue to its completion, and reads the card's copy-engine count.

usage: python copy_contention.py OUT.json   (needs the GPU to itself)
"""

import ctypes
import json
import statistics
import sys
import time

import torch

SMALL = 64 << 10
BIG = 1 << 30
REPEATS = 20


def device_attribute(attr):
    """cudaDeviceGetAttribute through the CUDA runtime torch has loaded (dlopen by soname finds the loaded copy)."""
    value = ctypes.c_int()
    try:
        rc = ctypes.CDLL("libcudart.so.13").cudaDeviceGetAttribute(ctypes.byref(value), attr, 0)
    except OSError as e:
        return f"unavailable: {e}"
    return value.value if rc == 0 else f"error {rc}"


def small_copy_ms(small_host, small_dev, stream, after=None):
    """Enqueue the small copy (after `after`'s large copy, if any) and time it on the host from enqueue to completion."""
    done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        start = time.perf_counter()
        small_dev.copy_(small_host, non_blocking=True)
        done.record()
    done.synchronize()
    return (time.perf_counter() - start) * 1e3


def main():
    out = sys.argv[1]
    torch.cuda.init()
    small_host = torch.empty(SMALL, dtype=torch.uint8).pin_memory()
    small_dev = torch.empty(SMALL, dtype=torch.uint8, device="cuda")
    big_host = torch.empty(BIG, dtype=torch.uint8).pin_memory()
    big_dev = torch.empty(BIG, dtype=torch.uint8, device="cuda")
    compute, offload = torch.cuda.Stream(), torch.cuda.Stream()

    def big_copy(direction):
        """Start a 1 GiB copy on the offload stream; return the event that marks its end."""
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(offload):
            if direction == "h2d":
                big_dev.copy_(big_host, non_blocking=True)
            else:
                big_host.copy_(big_dev, non_blocking=True)
            end.record()
        return end

    for _ in range(3):  # warm up both directions
        big_copy("h2d").synchronize()
        big_copy("d2h").synchronize()
        small_copy_ms(small_host, small_dev, compute)

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "async_engine_count": device_attribute(40),  # cudaDevAttrAsyncEngineCount
        "can_use_host_pointer_for_registered_mem": device_attribute(91),
        "small_bytes": SMALL,
        "big_bytes": BIG,
        "repeats": REPEATS,
    }
    alone = [small_copy_ms(small_host, small_dev, compute) for _ in range(REPEATS)]
    result["small_alone_ms"] = {"median": statistics.median(alone), "max": max(alone)}
    for direction in ("h2d", "d2h"):
        beside, big_ms = [], []
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            begin.record(offload)
            end = big_copy(direction)
            beside.append(small_copy_ms(small_host, small_dev, compute))
            end.synchronize()
            big_ms.append(begin.elapsed_time(end))
        result[f"small_beside_big_{direction}_ms"] = {
            "median": statistics.median(beside),
            "max": max(beside),
        }
        big_median = statistics.median(big_ms)
        result[f"big_{direction}_ms"] = {"median": big_median, "gb_per_s": BIG / big_median / 1e6}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
