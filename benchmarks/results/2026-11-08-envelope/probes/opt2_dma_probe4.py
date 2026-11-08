"""opt2_dma_probe4.py: before opt2_copy's E2 points, on the bare card: the SM copy kernel with 4 programs, as opt2_copy
installs it (vLLM's partial, bytes_per_chunk 8192, 32 KiB pages scattered on the card), copies 1.46 GB of random bytes
exactly as the copy engine does, and its time beside the copy engine's."""
import functools

import numpy as np
import torch
import vllm._custom_ops  # noqa: F401
import vllm.v1.kv_offload.cpu.swap_blocks_triton as st

PAGE, N = 32768, int(1.46e9) // 32768
host = torch.randint(0, 256, (N * PAGE,), dtype=torch.uint8).pin_memory()
a = torch.zeros(N * PAGE, dtype=torch.uint8, device="cuda")
b = torch.zeros_like(a)
slots = torch.from_numpy(np.random.default_rng(1).permutation(N).astype(np.int64))
src = host.data_ptr() + torch.arange(N, dtype=torch.int64) * PAGE
sizes = torch.full((N,), PAGE, dtype=torch.int64)
st.NUM_SMS = 4
copy = functools.partial(st.swap_blocks_batch, bytes_per_chunk=8192)
copy(src[:64], a.data_ptr() + slots[:64] * PAGE, sizes[:64], is_src_access_order_any=True)  # compile
torch.cuda.synchronize()
times = {}
for name, fn, dst in (("sm4", copy, a), ("dma", torch.ops._C_cache_ops.swap_blocks_batch, b)):
    dst.zero_()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    fn(src, dst.data_ptr() + slots * PAGE, sizes, True) if name == "dma" else fn(src, dst.data_ptr() + slots * PAGE, sizes, is_src_access_order_any=True)
    end.record()
    end.synchronize()
    times[name] = begin.elapsed_time(end)
same = torch.equal(a, b)
expected = torch.equal(a.view(N, PAGE)[slots[:4096].cuda()].cpu(), host.view(N, PAGE)[:4096])
print(f"1.46 GB in {N} pages: SM kernel 4 programs {times['sm4']:.0f} ms, copy engine {times['dma']:.0f} ms; same bytes: {same}; first 4,096 pages match the host: {expected}")
