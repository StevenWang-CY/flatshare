"""bench_marlin.py: Qwen3-4B-Instruct-2507-FP8's linear layers on this GPU, kernel against kernel (docs/HARDWARE.md, the
Marlin dry run, session 2 of the Marlin dry run): the GPU time of one layer's GEMM for M tokens, as vLLM 0.32.0 runs
it, - Marlin: the FP8 checkpoint's block-scaled weights through MarlinFP8ScaledMMLinearKernel's path, W8A16
(process_fp8_weight_block_strategy, prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear: vLLM's marlin_gemm), a
layer's calls sharing one lock workspace as the engine's do (marlin_make_workspace_new, once); - BF16: torch's GEMM on
the same weights in BF16, the rate Marlin's factor is taken against; - DeepGEMM: DeepGemmFp8BlockScaledMMKernel's
path, W8A8 (the activations' per-token-group FP8 quantization and fp8_gemm_nt), where the card runs it (sm90, sm100,
sm120; the box's own path); at the model's four shapes, qkv 2,560 to 6,144, o 4,096 to 2,560, gate_up 2,560 to 19,456
and down 9,728 to 2,560, for each M of --m.

Each time is one call's GPU time as the call runs inside a CUDA graph, which is where vLLM runs these layers: it
captures every step of up to 512 tokens by default, the box's step cap. INNER calls, each after an L2 flush (a buffer
twice the L2's size written), are captured in one graph and INNER flushes in another; each graph is replayed --iters
times between two CUDA events, and one call's time is the calls' median replay less the flushes', over INNER. One
call timed alone between two events would measure instead the host's dispatch of its kernels, which the GPU waits
for. DeepGEMM JIT-compiles its kernels with nvcc at their first call, so the bench runs in the engine unit's
environment (its CUDA_HOME and PATH; deploy/a100/host.py with-engine-env NODE CMD...); a DeepGEMM row whose build
failed without CUDA_HOME says so. On the box, from the home directory, with the engine stopped:

    leaseholder/venvs/engine/bin/python leaseholder/src/deploy/a100/host.py with-engine-env \\
      leaseholder/src/deploy/phastform leaseholder/venvs/engine/bin/python \\
      leaseholder/src/benchmarks/a100/bench_marlin.py --m 1,16,64,128,256,512,1024 --out gemm.csv

The CSV has one row per shape, M and kernel: the time in µs, the TFLOPS it amounts to, how it was timed (cuda-graph)
and, for a kernel that does not run here, why, in one line (the row's time is empty). Its last lines on stdout are
each M's Marlin-over-BF16 ratio summed over the shapes: the box's data point beside PR #61224's 1.42 on an L4, which
does not say how it timed its GEMMs. The module imports without a GPU, vLLM or torch (the tests' CPU, which drives it
with a fake torch.cuda), which the GPU work imports when it runs.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

SHAPES = (("qkv", 6144, 2560), ("o", 2560, 4096), ("gate_up", 19456, 2560), ("down", 2560, 9728))
"""Each linear layer's name and its weight's (N, K), out by in features (the model's config.json)."""
KERNELS = ("marlin", "bf16", "deepgemm")
BLOCK = 128  # the checkpoint's FP8 weight blocks, 128 x 128
INNER = 10
"""The calls of a kernel, each after an L2 flush, that one CUDA graph holds."""
TIMING = "cuda-graph"  # how a row's time was taken, in the CSV
NOTE = 400  # a note's most characters


@dataclass
class Row:
    shape: str
    m: int
    kernel: str
    us: float | None
    note: str = ""

    def tflops(self, n: int, k: int) -> float | None:
        return None if not self.us else 2 * self.m * n * k / (self.us * 1e-6) / 1e12


def parse_ms(text: str) -> list[int]:
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(
            f"--m takes positive token counts, comma separated: {text!r}"
        )
    return values


def write_csv(rows: Sequence[Row], path: Path) -> None:
    sizes = {name: (n, k) for name, n, k in SHAPES}
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["shape", "n", "k", "m", "kernel", "us", "tflops", "timing", "note"])
        for row in rows:
            n, k = sizes[row.shape]
            tflops = row.tflops(n, k)
            out.writerow(
                [
                    row.shape,
                    n,
                    k,
                    row.m,
                    row.kernel,
                    "" if row.us is None else f"{row.us:.2f}",
                    "" if tflops is None else f"{tflops:.1f}",
                    "" if row.us is None else TIMING,
                    row.note,
                ]
            )


def ratios(rows: Sequence[Row]) -> dict[int, float]:
    """Each M's Marlin time over BF16's, summed over the shapes (where both ran)."""
    out = {}
    for m in sorted({row.m for row in rows}):
        times = {
            k: [r.us for r in rows if r.m == m and r.kernel == k and r.us]
            for k in ("marlin", "bf16")
        }
        if times["marlin"] and len(times["marlin"]) == len(times["bf16"]):
            out[m] = sum(times["marlin"]) / sum(times["bf16"])
    return out


def per_call_us(pairs_ms: Sequence[float], flushes_ms: Sequence[float]) -> float:
    """One call's GPU time in µs from the replays, in ms, of a graph of INNER flushes and calls and of a graph of
    INNER flushes: the medians' difference over INNER."""
    us = (statistics.median(pairs_ms) - statistics.median(flushes_ms)) * 1000.0 / INNER
    if us <= 0:
        raise RuntimeError(f"the call's time, {us:.2f} µs, is within the flushes' noise")
    return us


def timed(fn: Callable[[], object], iters: int, flush: Callable[[], None]) -> float:
    """The GPU time of one call of `fn` in µs, after an L2 flush, as it runs inside a CUDA graph: warmed up on a side
    stream, then INNER flushes and calls captured in one graph and INNER flushes in another, each replayed `iters`
    times between two CUDA events (per_call_us). The GPU never waits on Python between the calls."""
    import torch

    cuda = torch.cuda
    side = cuda.Stream()
    side.wait_stream(cuda.current_stream())
    with cuda.stream(side):  # JIT builds, autotuning and first allocations, before any capture
        for _ in range(3):
            flush()
            fn()
    cuda.current_stream().wait_stream(side)
    cuda.synchronize()
    pairs, flushes = cuda.CUDAGraph(), cuda.CUDAGraph()
    with cuda.graph(pairs):
        for _ in range(INNER):
            flush()
            fn()
    with cuda.graph(flushes):
        for _ in range(INNER):
            flush()

    def replays(graph) -> list[float]:
        samples = []
        for _ in range(iters):
            start, end = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            cuda.synchronize()
            samples.append(start.elapsed_time(end))
        return samples

    return per_call_us(replays(pairs), replays(flushes))


def note(error: BaseException, kernel: str) -> str:
    """Why a kernel did not run, in one line: the error's first line; for a DeepGEMM build that failed, the nvcc it
    took and the compiler's error, and, without CUDA_HOME (DeepGEMM's JIT then takes /usr/local/cuda's nvcc, on the
    box older than its sm120 kernels need), that the bench needs the engine unit's environment."""
    text = str(error)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = f"{type(error).__name__}: {lines[0] if lines else ''}"
    if kernel != "deepgemm" or "nvcc" not in text:
        return first[:NOTE]
    found = re.search(r"\S*nvcc(?=\s|$)", text)
    nvcc = found.group(0) if found else "nvcc"
    cause = next((line[line.index("error:") :] for line in lines[1:] if "error:" in line), first)
    if os.environ.get("CUDA_HOME"):
        return f"DeepGEMM's build with {nvcc} failed: {cause}"[:NOTE]
    return (
        "DeepGEMM needs nvcc: set CUDA_HOME (the engine unit's environment: deploy/a100/host.py with-engine-env); "
        f"its build with {nvcc} failed: {cause}"
    )[:NOTE]


def kernels(n: int, k: int) -> dict[str, Callable[[object], object] | str]:
    """Each kernel's GEMM for one (N, K) weight of random FP8 values and block scales, as a function of the
    activations, or why it does not run here."""
    import torch
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        MARLIN_MAX_BLOCKS_PER_SM,
        marlin_make_workspace_new,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )

    device = torch.device("cuda")
    weight = (torch.randn(n, k, device=device) * 0.05).to(torch.float8_e4m3fn)
    scale = (
        torch.rand(-(-n // BLOCK), -(-k // BLOCK), device=device, dtype=torch.float32) * 0.01
        + 0.001
    )
    found: dict[str, Callable[[object], object] | str] = {}

    layer = torch.nn.Module()
    w, s = process_fp8_weight_block_strategy(weight.clone(), scale.clone())
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(s, requires_grad=False)
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.weight_block_size, layer.orig_dtype, layer.bias = [BLOCK, BLOCK], torch.bfloat16, None
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    # The engine's Marlin calls on a stream share one zeroed lock buffer, MARLIN_MAX_BLOCKS_PER_SM an SM, which each
    # call leaves zeroed (marlin_utils.get_marlin_workspace, from vLLM's workspace manager); outside the engine,
    # workspace=None allocates and zeroes one a call.
    workspace = marlin_make_workspace_new(weight.device, MARLIN_MAX_BLOCKS_PER_SM)
    found["marlin"] = lambda x: apply_fp8_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale_inv,
        workspace=workspace,
        size_n=n,
        size_k=k,
        bias=None,
    )

    dense = weight.to(torch.bfloat16) * scale.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(
        BLOCK, 1
    )[:, :k].to(torch.bfloat16)
    found["bf16"] = lambda x: torch.nn.functional.linear(x, dense)

    try:
        import vllm.envs as envs
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            deepgemm_post_process_fp8_weight_block,
            per_token_group_quant_fp8,
        )
        from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used, is_deep_gemm_supported

        if not is_deep_gemm_supported():
            raise RuntimeError("DeepGEMM runs on Hopper and Blackwell GPUs only")
        e8m0 = is_deep_gemm_e8m0_used()
        dg_weight, dg_scale = deepgemm_post_process_fp8_weight_block(
            wq=weight.clone(),
            ws=scale.clone(),
            quant_block_shape=(BLOCK, BLOCK),
            use_e8m0=e8m0,
            is_bmm=False,
            bmm_batch_size=0,
        )

        def deepgemm(x):
            q, xs = per_token_group_quant_fp8(
                x,
                BLOCK,
                use_ue8m0=e8m0,
                column_major_scales=True,
                tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES,
            )
            out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
            fp8_gemm_nt((q, xs), (dg_weight, dg_scale), out, is_deep_gemm_e8m0_used=e8m0)
            return out

        found["deepgemm"] = deepgemm
    except Exception as e:  # not this card, or not this vLLM: the row says why
        found["deepgemm"] = note(e, "deepgemm")
    return found


def run(ms: Sequence[int], iters: int) -> list[Row]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("bench_marlin: no CUDA device")
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    buffer = torch.empty(2 * l2 // 4 + 1, dtype=torch.float32, device="cuda")

    def flush() -> None:
        buffer.zero_()

    rows = []
    for name, n, k in SHAPES:
        found = kernels(n, k)
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            for kernel in KERNELS:
                fn = found[kernel]
                if isinstance(fn, str):
                    rows.append(Row(name, m, kernel, None, fn))
                    continue
                try:
                    rows.append(Row(name, m, kernel, timed(lambda fn=fn, x=x: fn(x), iters, flush)))
                except Exception as e:
                    rows.append(Row(name, m, kernel, None, note(e, kernel)))
        del found
        torch.cuda.empty_cache()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench_marlin.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--m", type=parse_ms, default=parse_ms("1,16,64,128,256,512,1024"))
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help=f"each graph's replays a cell, each of {INNER} calls (default %(default)s)",
    )
    parser.add_argument("--out", type=Path, required=True, help="the CSV to write")
    args = parser.parse_args(argv)
    if args.iters < 1:
        parser.error("--iters takes a positive count of replays")
    rows = run(args.m, args.iters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out)
    for m, ratio in ratios(rows).items():
        print(f"M {m}: Marlin over BF16 {ratio:.2f}")
    print(f"bench_marlin: {len(rows)} rows in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
