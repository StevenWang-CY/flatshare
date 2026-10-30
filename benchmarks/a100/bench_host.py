"""bench_host.py: the host around the card, the two of its costs the forecast takes from the box (docs/BENCHMARKS.md,
the analysis): the link's copy rate between pinned host memory and this GPU, each way, and the CPU's speed at Python.

The link. vLLM 0.32.0's CPU tier moves KV in pages, one page a layer and block (K and V apart in the LBHNC layout the
connector requires: 32 KiB for Qwen3-4B's 16-token block of 8 KV heads of 128 at BF16), all of a transfer's pages in one
driver call (vllm._custom_ops.swap_blocks_batch: cuMemcpyBatchAsync on CUDA 12.8 and later), host-to-device with the
source's access order relaxed and device-to-host in stream order, as its gpu_worker submits them. For each direction
and each page size of PAGES, --total-mib of pages is copied through that op, between two CUDA events on a stream of
its own, --reps times after one warm-up; and the same bytes as one copy (torch's copy_), the link's own rate. A row's
rate is the bytes over the median time, in GB/s (10^9 bytes a second, the unit PCIe's rates are given in). The
restore's per-token term in E0 (µs a token, at 144 KiB of KV a token) is read against these rates: the share of the
link a restore reaches, and whether the link or the copies' submission bounds it.

The CPU. One fixed pure-Python workload, the kind of work vLLM's scheduler and input preparation do in each step
(a priority heap of requests, dict and attribute updates, list building), timed --reps times on one core; its median
in ms is the host's Python time, read as a ratio between hosts (the box's step overhead, which the forecast keeps,
is Python on the box's CPU). It runs without a GPU (--no-gpu), so that any host gives its number.

The facts beside them: the CPU's model name, the GPU, its PCIe generation and width as nvidia-smi gives them (on a
VM, the link to the GPU's switch, which need not be the root port's), torch's and vLLM's versions. On the host, from
the home directory, with the GPU units stopped:

    flatshare/venvs/engine/bin/python flatshare/src/benchmarks/a100/bench_host.py --out flatshare/runs/a100/host

writes host.json there and prints a line for each row. The module imports without a GPU, vLLM or torch (the tests'
CPU), which the link's measurement imports when it runs.
"""

from __future__ import annotations

import argparse
import heapq
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PAGES = (32 * 1024, 64 * 1024, 16 * 144 * 1024)
"""The page sizes copied, in bytes: one layer's K or V of a 16-token block (the engine's page), K and V together, and a
whole block of all 36 layers (144 KiB of KV a token)."""
DIRECTIONS = ("h2d", "d2h")


@dataclass
class Row:
    direction: str
    page_bytes: int | None  # None: the bytes as one copy
    pages: int
    bytes: int
    ms: float
    note: str = ""

    @property
    def gb_per_s(self) -> float | None:
        return self.bytes / (self.ms * 1e-3) / 1e9 if self.ms > 0 else None


def python_work() -> int:
    """A fixed workload of the scheduler's kind: 8,000 requests through a priority heap, each one's state updated in a
    dict and on an object, and a step's list rebuilt from the running 256 each time one starts; deterministic."""

    class Request:
        __slots__ = ("arrival", "priority", "rid", "state", "tokens")

        def __init__(self, rid: int) -> None:
            self.rid = rid
            self.priority = rid % 3
            self.arrival = (rid * 7919) % 10007
            self.tokens = 0
            self.state = "waiting"

    requests = [Request(i) for i in range(8_000)]
    waiting = [(r.priority, r.arrival, r.rid) for r in requests]
    heapq.heapify(waiting)
    running: dict[int, Request] = {}
    total = 0
    while waiting:
        _, _, rid = heapq.heappop(waiting)
        request = requests[rid]
        request.state = "running"
        running[rid] = request
        if len(running) > 256:  # a step's batch: decode one token for each, finish the oldest
            batch = [r for r in running.values() if r.state == "running"]
            for r in batch:
                r.tokens += 1
            oldest = min(running)
            running.pop(oldest).state = "finished"
            total += len(batch)
    return total + sum(r.tokens for r in requests)


def python_ms(reps: int) -> list[float]:
    """The workload's times on this core, in ms, after one warm-up."""
    python_work()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        python_work()
        times.append((time.perf_counter() - start) * 1e3)
    return times


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_facts() -> dict[str, str]:
    fields = (
        "name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}"[:200]}
    values = [v.strip() for v in out.splitlines()[0].split(",")] if out.strip() else []
    return dict(zip(fields.split(","), values, strict=False))


def link_rows(total_mib: int, reps: int) -> list[Row]:
    """The link's rows on this GPU (see the module's docstring)."""
    import torch
    from vllm import _custom_ops as ops

    total = total_mib * 1024 * 1024
    host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    device = torch.empty(total, dtype=torch.uint8, device="cuda")
    host.fill_(1)
    device.fill_(2)
    stream = torch.cuda.Stream()
    rows = []

    def timed(copy) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for rep in range(reps + 1):
            with torch.cuda.stream(stream):
                start.record(stream)
                copy()
                end.record(stream)
            end.synchronize()
            if rep:  # the first is the warm-up
                times.append(start.elapsed_time(end))
        return statistics.median(times)

    for direction in DIRECTIONS:
        src, dst = (host, device) if direction == "h2d" else (device, host)
        for page in PAGES:
            pages = total // page
            offsets = torch.arange(pages, dtype=torch.int64) * page
            src_ptrs, dst_ptrs = offsets + src.data_ptr(), offsets + dst.data_ptr()
            sizes = torch.full((pages,), page, dtype=torch.int64)
            relaxed = (
                direction == "h2d"
            )  # gpu_worker: loads relax the source's order, stores keep the stream's

            def batch(src_ptrs=src_ptrs, dst_ptrs=dst_ptrs, sizes=sizes, relaxed=relaxed) -> None:
                ops.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, is_src_access_order_any=relaxed)

            rows.append(Row(direction, page, pages, pages * page, timed(batch)))
        rows.append(
            Row(
                direction, None, 1, total, timed(lambda d=dst, s=src: d.copy_(s, non_blocking=True))
            )
        )
    return rows


def describe(row: Row) -> str:
    size = (
        "one copy"
        if row.page_bytes is None
        else f"{row.pages:,} pages of {row.page_bytes // 1024:,} KiB"
    )
    rate = f"{row.gb_per_s:.2f} GB/s" if row.gb_per_s else "no rate"
    return f"link {row.direction}: {row.bytes / 2**20:,.0f} MiB as {size}: {row.ms:.2f} ms, {rate}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the directory host.json goes to")
    parser.add_argument(
        "--total-mib", type=int, default=256, help="the bytes each row copies (MiB)"
    )
    parser.add_argument(
        "--reps", type=int, default=7, help="each measurement's repetitions, after one warm-up"
    )
    parser.add_argument("--no-gpu", action="store_true", help="the CPU's number alone")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    record: dict = {"cpu": cpu_model(), "python": platform.python_version()}
    times = python_ms(args.reps)
    record["python_ms"] = {"p50": statistics.median(times), "min": min(times), "reps": len(times)}
    print(
        f"python: {record['python_ms']['p50']:.1f} ms at the median of {len(times)} on {record['cpu']}"
    )
    if not args.no_gpu:
        record["gpu"] = gpu_facts()
        try:
            import torch
            import vllm

            record["versions"] = {"torch": torch.__version__, "vllm": vllm.__version__}
            rows = link_rows(args.total_mib, args.reps)
        except Exception as error:  # the CPU's number stands without the link's
            record["link_error"] = f"{type(error).__name__}: {error}"[:400]
            print(f"link: {record['link_error']}", file=sys.stderr)
            rows = []
        record["link"] = [asdict(row) | {"gb_per_s": row.gb_per_s} for row in rows]
        for row in rows:
            print(describe(row))
    (args.out / "host.json").write_text(json.dumps(record, indent=2) + "\n")
    return 1 if record.get("link_error") else 0


if __name__ == "__main__":
    sys.exit(main())
