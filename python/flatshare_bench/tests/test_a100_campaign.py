"""The A100 run's campaign files (benchmarks/a100): the core's five rounds and the optional arms' campaigns name arm
files that load, with the seeds and the order docs/HARDWARE.md, the core block by block gives; probe_results.py reads what probe.py and tail.py
print and picks C''s stock arm by P-1's rule; and the run scripts parse."""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from flatshare_bench.e2.arm import load_arm

REPO = Path(__file__).resolve().parents[3]
A100 = REPO / "benchmarks" / "a100"
DEPLOY = REPO / "deploy"
FAKES = Path(__file__).parent / "fakes"


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


results = _module("flatshare_a100_probe_results", A100 / "probe_results.py")
PR = "make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48"


def campaign(path: Path) -> tuple[int | None, list[Path]]:
    """A campaign file's agents seed and its arm files, as campaign.sh reads them."""
    seed, arms = None, []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if m := re.fullmatch(r"agents-seed\s+(\d+)", line):
            seed = int(m.group(1))
        elif not re.fullmatch(r"kv-sample-s\s+[\d.]+", line):
            arms.append(path.parent / line)
    return seed, arms


def sampling(path: Path) -> str | None:
    """A campaign file's KV sampler interval line's value, if it has one."""
    found = re.findall(r"^kv-sample-s\s+([\d.]+)", path.read_text(), re.M)
    return found[0] if found else None


def test_the_marlin_point_runs_as_w5s_did_and_the_core_samples() -> None:
    # The Marlin point is read against W5's, which ran without the KV sampler; the A100's own points sample it,
    # for the running share its analysis reads.
    assert sampling(A100 / "marlin-dry.campaign") == "0"
    for path in A100.glob("a100-*.campaign"):
        assert sampling(path) is None, path.name


def test_the_cores_rounds_rotate_the_engines_with_no_switch_between_rounds() -> None:
    previous = PR  # block 3 leaves the PRs' values running
    w5 = {"w5-disc-pr-ab48-s24.toml", "w5-disc-frozen-s24.toml", "w5-a1-voice.toml"}
    for n in range(1, 6):
        seed, arms = campaign(A100 / f"a100-r{n}.campaign")
        assert seed == 10 + n
        assert {arm.name for arm in arms} == w5 and all(arm.parent.name == "arms" for arm in arms)
        engines = [str(load_arm(arm).engine) for arm in arms]
        assert engines[0] == previous, n
        assert engines.count(PR) == 1 and engines.count("frozen") == 2
        # The engine switches once within a round: voice alone sits beside the frozen point.
        assert sum(a != b for a, b in itertools.pairwise(engines)) == 1
        previous = engines[-1]


@pytest.mark.parametrize("path", sorted(A100.glob("*.campaign")), ids=lambda p: p.name)
def test_every_campaign_names_arm_files_that_load(path: Path) -> None:
    seed, arms = campaign(path)
    assert arms and seed is not None
    for arm in arms:
        assert arm.is_file(), arm
        load_arm(arm)


def test_a_prime_is_the_cores_arms_at_32_sessions() -> None:
    for ours, w5 in (
        ("a100-aprime-frozen-s32.toml", "w5-disc-frozen-s24.toml"),
        ("a100-aprime-pr-ab48-s32.toml", "w5-disc-pr-ab48-s24.toml"),
    ):
        a, b = load_arm(A100 / "arms" / ours), load_arm(REPO / "benchmarks" / "e2" / "arms" / w5)
        assert a.agents.sessions == 32 and b.agents.sessions == 24
        assert (a.engine, a.voice, a.agents.data, a.arm) == (
            b.engine,
            b.voice,
            b.agents.data,
            b.arm,
        )
    seeds = [campaign(A100 / f"a100-aprime-r{n}.campaign")[0] for n in (1, 2)]
    resubmit = [campaign(A100 / f"a100-resubmit-r{n}.campaign")[0] for n in (3, 4, 5)]
    assert seeds == [11, 12] and resubmit == [13, 14, 15]
    # Each round pairs the two engines on its sample; voice alone, the SLO's reference, is the core's five points on
    # the same host, which flatshare-bench e2 report reads for every point of that hardware.
    for n in (1, 2):
        arms = {arm.name for arm in campaign(A100 / f"a100-aprime-r{n}.campaign")[1]}
        assert arms == {"a100-aprime-frozen-s32.toml", "a100-aprime-pr-ab48-s32.toml"}, n


def test_tuned_vllm_is_the_cores_frozen_point_at_the_probes_best_cap_on_the_cores_samples() -> None:
    # The stock engine at the step cap blocks 1 and 2 found best on the A100 (vLLM's default, 2,048/0, as block 2's
    # stock2048 ran it), otherwise the core's frozen point, on the agents' samples of the core's rounds 1 and 2, so
    # that each pairs with that round's frozen point and the PRs' values.
    tuned = load_arm(A100 / "arms" / "a100-tuned-s24.toml")
    frozen = load_arm(REPO / "benchmarks" / "e2" / "arms" / "w5-disc-frozen-s24.toml")
    assert (
        str(tuned.engine) == "frozen --max-num-batched-tokens 2048 --long-prefill-token-threshold 0"
    )
    assert (tuned.voice, tuned.agents.sessions, tuned.agents.data) == (
        frozen.voice,
        frozen.agents.sessions,
        frozen.agents.data,
    )
    for n in (1, 2):
        seed, arms = campaign(A100 / f"a100-tuned-r{n}.campaign")
        assert seed == campaign(A100 / f"a100-r{n}.campaign")[0] == 10 + n
        assert [arm.name for arm in arms] == ["a100-tuned-s24.toml"], n


def test_the_prs_at_the_a100s_best_cap_are_the_cores_pr_point_at_tuned_vllms_cap() -> None:
    # The PRs' 48-token configuration on tuned vLLM's 2,048/0, otherwise the core's point, on the samples of the
    # core's rounds 1 and 2, so that each pairs with tuned vLLM's round and that round's two core points.
    ours = load_arm(A100 / "arms" / "a100-pr-ab48-2048-s24.toml")
    core = load_arm(REPO / "benchmarks" / "e2" / "arms" / "w5-disc-pr-ab48-s24.toml")
    tuned = load_arm(A100 / "arms" / "a100-tuned-s24.toml")
    assert str(ours.engine) == (
        "make-room --max-num-batched-tokens 2048 --long-prefill-token-threshold 0 "
        "--additional-config.make_room.reserve_tokens 0 "
        "--additional-config.make_room.budget_cap_tokens 48"
    )
    assert (
        str(ours.engine)
        .removeprefix("make-room ")
        .startswith(str(tuned.engine).removeprefix("frozen "))
    )
    assert (ours.arm, ours.voice, ours.agents.sessions, ours.agents.data) == (
        core.arm,
        core.voice,
        core.agents.sessions,
        core.agents.data,
    )
    assert "vLLM PRs A and B with a 48-token window" in ours.label
    for n in (1, 2):
        seed, arms = campaign(A100 / f"a100-prtuned-r{n}.campaign")
        assert seed == campaign(A100 / f"a100-tuned-r{n}.campaign")[0] == 10 + n
        assert [arm.name for arm in arms] == ["a100-pr-ab48-2048-s24.toml"], n


PROBE_LINE = (
    "kv: priority-0 TTFT {ttft} ms, 16 tokens {t16} ms, 40 tokens {t40}, for 761 prompt tokens, sent with 2 "
    "priority-2 requests of 17999 tokens running and 24 of 2274 blocks free; after it, 2 running and 0 waiting"
)
TAIL = """tail: 12 agents with fresh prompts of 6,000-30,000 tokens beside 2 calls with 40-token replies, for 240 s
replies: 40 turns, 16 tokens p50 {r16} p90 400 p95 420 max 500 ms, 40 tokens p50 {r40} p90 917 p95 2372 max ≥30000 ms
agents: 100 requests in 240 s, 0.42 a second, with 7,000 prompt and 20 output tokens a second
calls: 40 turns, {failed} failed (counted at 30,000 ms), TTFT p50 110 p90 300 p95 {p95} max 900 ms; agents: 0 of 12 stopped early
"""


def write_runs(root: Path, p95s: dict[str, list[str]]) -> None:
    for prefix, values in p95s.items():
        block = root / ("block1" if prefix == "frozen" else "block2")
        block.mkdir(parents=True, exist_ok=True)
        for i, p95 in enumerate(values, 1):
            text = TAIL.format(r16=300, r40=900, failed=int(p95.startswith("≥")), p95=p95)
            (block / f"{prefix}-tail-{i}.txt").write_text(text)


def test_the_probes_outputs_read_back() -> None:
    probe = results.parse_probe(
        "probe: ...\n" + PROBE_LINE.format(ttft=1650, t16="1800", t40="2400 ms")
    )
    assert (probe.case, probe.ttft_ms, probe.marks) == ("kv", 1650, {16: 1800, 40: 2400})
    assert (probe.running, probe.prompt_tokens, probe.free, probe.blocks) == (2, 17999, 24, 2274)
    ended = results.parse_probe(PROBE_LINE.format(ttft=120, t16="140", t40="—"))
    assert ended.marks == {16: 140, 40: None}
    tail = results.parse_tail(TAIL.format(r16=310, r40=910, failed=1, p95="≥30000"))
    assert (tail.turns, tail.failed, tail.stopped, tail.agents) == (40, 1, 0, 12)
    assert tail.ttft["p95"] == (30000, True) and tail.ttft["p50"] == (110, False)
    assert tail.replies[40]["max"] == (30000, True) and tail.replies[16]["p50"] == (310, False)
    assert tail.requests_per_s == pytest.approx(0.42)
    with pytest.raises(ValueError):
        results.parse_tail("tail: nothing measured\n")


def test_c_primes_stock_arm_is_the_lowest_median_first_token_p95(tmp_path: Path, capsys) -> None:
    write_runs(
        tmp_path,
        {
            "frozen": ["2400", "2600"],
            "stock2048": ["1900", "≥30000"],
            "stock1536-768": [
                "2000",
                "2100",
            ],  # the time-equivalent cap's runs: its cap and threshold
        },
    )
    name, flags, table = results.pick_stock(tmp_path, 1536)
    # 2,048/0's median is (1,900 + 30,000)/2: a failed turn counts at the timeout.
    assert name == "1536" and flags == (
        "--max-num-batched-tokens",
        "1536",
        "--long-prefill-token-threshold",
        "768",
    )
    assert [row[2] for row in table] == [2500, 15950, 2050]
    assert results.main(["pick-stock", str(tmp_path), "--cap", "1536"]) == 0
    assert (
        capsys.readouterr().out
        == "1536 --max-num-batched-tokens 1536 --long-prefill-token-threshold 768\n"
    )
    write_runs(tmp_path, {"frozen": ["1000", "1100"]})
    assert results.pick_stock(tmp_path, 1536)[:2] == ("frozen", ())
    # A setting without its tail runs is an error, not a silent pick of the others.
    assert results.main(["pick-stock", str(tmp_path), "--cap", "1280"]) == 2
    table = results.markdown(tmp_path)
    assert "| block1/frozen-tail-1 | tail (p50) | 110 | 300 | 900 | 1000 |" in table
    # At the forecast's low value, 2,048, the cap's own runs (2,048/1,024) stand beside vLLM's default (2,048/0).
    write_runs(tmp_path, {"frozen": ["2400", "2600"], "stock2048-1024": ["900", "1000"]})
    name, flags, table = results.pick_stock(tmp_path, 2048)
    assert name == "2048" and flags[-1] == "1024"
    assert [row[2] for row in table] == [2500, 15950, 950]


SCRIPTS = sorted(
    [
        *A100.glob("*.sh"),
        *(DEPLOY / "a100").glob("*.sh"),
        DEPLOY / "a100" / "requirements" / "compile.sh",
    ]
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_run_scripts_parse_and_refuse_a_wrong_argument(script: Path, tmp_path: Path) -> None:
    # Their runs are the CPU dry run's (test_a100_dry_run.py); here bash parses each, and each that takes arguments
    # stops at a wrong one before it touches the host.
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    if script.name in {"compile.sh", "pull.sh", "push.sh"}:
        return
    done = subprocess.run(
        ["bash", str(script), "--no-such-option"],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ | {"HOME": str(tmp_path)},
    )
    assert done.returncode == 2 and "usage" in done.stderr, (done.returncode, done.stderr)
    assert not any(tmp_path.iterdir()), sorted(tmp_path.rglob("*"))  # nothing written first


def test_bench_marlin_imports_without_a_gpu_and_writes_its_rows(tmp_path: Path) -> None:
    bench = _module("flatshare_a100_bench_marlin", A100 / "bench_marlin.py")
    # The model's four linear shapes, (N, K): qkv, o, gate_up and down of Qwen3-4B-Instruct-2507.
    assert bench.SHAPES == (
        ("qkv", 6144, 2560),
        ("o", 2560, 4096),
        ("gate_up", 19456, 2560),
        ("down", 2560, 9728),
    )
    assert bench.parse_ms("1,16,1024") == [1, 16, 1024]
    with pytest.raises(Exception, match="positive"):
        bench.parse_ms("0,16")
    rows = [
        bench.Row(shape, m, kernel, us)
        for shape, *_ in bench.SHAPES
        for m, kernel, us in ((16, "marlin", 30.0), (16, "bf16", 20.0), (16, "deepgemm", None))
    ]
    rows[2].note = "RuntimeError: DeepGEMM runs on Hopper and Blackwell GPUs only"
    assert bench.ratios(rows) == {16: 1.5}
    bench.write_csv(rows, tmp_path / "gemm.csv")
    lines = (tmp_path / "gemm.csv").read_text().splitlines()
    assert lines[0] == "shape,n,k,m,kernel,us,tflops,timing,note"
    # 2 x 16 x 6,144 x 2,560 FLOP in 30 µs, a call's GPU time in a CUDA graph.
    assert lines[1] == "qkv,6144,2560,16,marlin,30.00,16.8,cuda-graph,"
    assert lines[3].startswith("qkv,6144,2560,16,deepgemm,,,,RuntimeError")


def test_bench_host_gives_the_cpus_number_without_a_gpu_and_reads_a_rate_in_gb_per_s(
    tmp_path: Path,
) -> None:
    bench = _module("flatshare_a100_bench_host", A100 / "bench_host.py")
    # The engine's page, one layer's K or V of a 16-token block (8 heads of 128 at BF16), and a block of all 36 layers.
    assert bench.PAGES == (32 * 1024, 64 * 1024, 16 * 144 * 1024)
    assert bench.python_work() == bench.python_work()  # the same work on every host
    assert bench.main(["--no-gpu", "--reps", "2", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "host.json").read_text())
    assert record["python_ms"]["reps"] == 2 and record["python_ms"]["p50"] > 0
    assert "link" not in record and "gpu" not in record
    # 256 MiB in 20 ms is 13.42 GB/s, decimal, as PCIe's rates are given.
    row = bench.Row("h2d", 32 * 1024, 8192, 256 * 2**20, 20.0)
    assert round(row.gb_per_s, 2) == 13.42
    assert bench.describe(row) == "link h2d: 256 MiB as 8,192 pages of 32 KiB: 20.00 ms, 13.42 GB/s"
    assert bench.describe(bench.Row("d2h", None, 1, 256 * 2**20, 20.0)).startswith(
        "link d2h: 256 MiB as one copy:"
    )


class _Cuda:
    """torch.cuda for bench_marlin on a CPU, with two clocks in µs: the host's, which each kernel launch outside a
    graph advances by its Python dispatch, and the stream's, which runs a kernel once it is free and the launch has
    reached it, as a GPU's does. Inside `graph` a launch is captured, not run; a replay runs the captured kernels back to
    back for the cost of one launch. The run's third replay stalls 900 µs, a hiccup the median absorbs and a mean would
    not."""

    def __init__(self) -> None:
        self.host = self.gpu = 0.0
        self.capturing: _Graph | None = None
        self.on = "default"  # the stream launches go to
        self.ran: list[tuple[str, str]] = []  # each kernel run outside a graph, and its stream
        self.graphs: list[_Graph] = []
        self.replays = 0

    def launch(self, kernel: str, gpu_us: float, host_us: float) -> None:
        if self.capturing is not None:
            self.capturing.kernels.append((kernel, gpu_us))
            return
        self.host += host_us
        self.gpu = max(self.gpu, self.host) + gpu_us
        self.ran.append((kernel, self.on))

    def is_available(self) -> bool:
        return True

    def get_device_properties(self, index: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(L2_cache_size=32 << 20)

    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        self.host = max(self.host, self.gpu)

    def Stream(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(name="side", wait_stream=lambda other: None)

    def current_stream(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(name=self.on, wait_stream=lambda other: None)

    @contextlib.contextmanager
    def stream(self, stream: types.SimpleNamespace):
        before, self.on = self.on, stream.name
        try:
            yield
        finally:
            self.on = before

    def CUDAGraph(self) -> _Graph:
        graph = _Graph(self)
        self.graphs.append(graph)
        return graph

    @contextlib.contextmanager
    def graph(self, graph: _Graph):
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None

    def Event(self, enable_timing: bool = False) -> _Event:
        return _Event(self)


class _Graph:
    def __init__(self, cuda: _Cuda) -> None:
        self.cuda, self.kernels, self.replayed = cuda, [], 0

    def replay(self) -> None:
        cuda = self.cuda
        cuda.replays += 1
        self.replayed += 1
        cuda.host += 2.0
        stall = 900.0 if cuda.replays == 3 else 0.0
        cuda.gpu = max(cuda.gpu, cuda.host) + stall + sum(us for _, us in self.kernels)


class _Event:
    def __init__(self, cuda: _Cuda) -> None:
        self.cuda, self.at = cuda, 0.0

    def record(self) -> None:
        # Reached once the stream has run what was launched before it.
        self.cuda.host += 1.0
        self.cuda.gpu = self.at = max(self.cuda.gpu, self.cuda.host)

    def elapsed_time(self, end: _Event) -> float:
        return (end.at - self.at) / 1000.0  # ms, as torch's


def test_bench_marlin_times_a_call_as_it_runs_in_a_cuda_graph_not_its_dispatch_from_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _module("flatshare_a100_bench_marlin", A100 / "bench_marlin.py")
    cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))

    # A call of 40 µs on the GPU whose Python takes 400 µs to launch it, after an L2 flush of 300 µs launched in 5:
    # timed alone between two events, the call reads as its dispatch; inside vLLM's CUDA graphs it costs its 40 µs.
    def call() -> None:
        cuda.launch("call", 40.0, 400.0)

    def flush() -> None:
        cuda.launch("flush", 300.0, 5.0)

    assert bench.timed(call, 7, flush) == pytest.approx(40.0)
    # Warmed up first on a side stream, outside any graph (the JIT builds, autotuning and first allocations), then
    # INNER flushes and calls captured in one graph and INNER flushes in another, each replayed 7 times.
    assert cuda.ran == [("flush", "side"), ("call", "side")] * 3
    pairs, flushes = cuda.graphs
    assert pairs.kernels == [("flush", 300.0), ("call", 40.0)] * bench.INNER
    assert flushes.kernels == [("flush", 300.0)] * bench.INNER
    assert pairs.replayed == flushes.replayed == 7
    # A call that adds nothing to the flushes is within their noise, not a time.
    with pytest.raises(RuntimeError, match="noise"):
        bench.timed(lambda: None, 7, flush)


def _deep_jit_failure(nvcc: str) -> str:
    """DeepGEMM's error when its JIT build fails, as session 2 recorded it on the box without CUDA_HOME, abridged."""
    return (
        "Assertion error (third-party/deep_jit/include/deep_jit/utils/command.hpp:35): exit_code == 0: command "
        "failed with exit code 1:\n"
        f"cd /home/u/.cache/vllm/deep_gemm/tmp/1-2 && {nvcc} /home/u/.cache/vllm/deep_gemm/tmp/1-2/kernel.cu "
        "--cubin --gpu-architecture=sm_120f -O3 2>&1\n"
        "In file included from /home/u/venv/vllm/third_party/deep_gemm/include/deep_gemm/impls/"
        "sm120_fp8_fp4_gemm_1d1d.cuh:16,\n"
        "/home/u/venv/vllm/third_party/deep_gemm/include/deep_gemm/common/sm120_utils.cuh:28:2: "
        'error: #error "DeepGEMM SM120 kernels require CUDA 13.0 or newer"\n'
        '   28 | #error "DeepGEMM SM120 kernels require CUDA 13.0 or newer"\n'
        "  #13 /home/u/venv/torch/lib/libtorch_python.so+0x3f2bb1 pybind11::cpp_function::dispatcher(_object*, "
        "_object* const*, unsigned long, _object*) at ??:0\n"
    )


def _fake_kernels(monkeypatch: pytest.MonkeyPatch, cuda: _Cuda) -> tuple[list, list]:
    """torch and the vLLM 0.32.0 functions bench_marlin calls, on `cuda`: Marlin 40 µs a call (400 µs of Python),
    BF16 20, DeepGEMM's quantization 3 and its GEMM 25, which, as on the box, fails to build without CUDA_HOME (nvcc
    from /usr/local/cuda, older than its sm120 kernels need). Returns the workspaces marlin_make_workspace_new made,
    as (device, blocks per SM), and the workspace each Marlin call was given."""
    torch = mock.MagicMock(name="torch")
    torch.cuda = cuda
    torch.nn.Module = types.SimpleNamespace
    torch.nn.Parameter = lambda data, requires_grad=True: data
    torch.empty = lambda *args, **kwargs: types.SimpleNamespace(
        zero_=lambda: cuda.launch("flush", 300.0, 5.0)
    )
    torch.nn.functional.linear = lambda x, w: cuda.launch("bf16", 20.0, 30.0)
    monkeypatch.setitem(sys.modules, "torch", torch)
    made, given = [], []

    def make_workspace(device: object, max_blocks_per_sm: int = 1) -> object:
        made.append((device, max_blocks_per_sm))
        return f"workspace {len(made)}"

    def apply_marlin(*, input, weight, weight_scale, workspace, size_n, size_k, bias, **rest):
        given.append(workspace)
        cuda.launch("marlin", 40.0, 400.0)

    def quantize(x, block, **rest):
        cuda.launch("quant", 3.0, 40.0)
        return "q", "scales"

    def fp8_gemm_nt(a, b, out, is_deep_gemm_e8m0_used):
        if not os.environ.get("CUDA_HOME"):
            raise RuntimeError(_deep_jit_failure("/usr/local/cuda/bin/nvcc"))
        cuda.launch("deepgemm", 25.0, 60.0)

    utils = "vllm.model_executor.layers.quantization.utils"
    modules = {
        "vllm": {},
        "vllm.envs": {"VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES": True},
        "vllm.model_executor": {},
        "vllm.model_executor.layers": {},
        "vllm.model_executor.layers.quantization": {},
        utils: {},
        f"{utils}.fp8_utils": {
            "process_fp8_weight_block_strategy": lambda w, s: (w, s),
            "deepgemm_post_process_fp8_weight_block": lambda **kw: (kw["wq"], kw["ws"]),
            "per_token_group_quant_fp8": quantize,
        },
        f"{utils}.marlin_utils": {
            "MARLIN_MAX_BLOCKS_PER_SM": 4,
            "marlin_make_workspace_new": make_workspace,
        },
        f"{utils}.marlin_utils_fp8": {
            "apply_fp8_marlin_linear": apply_marlin,
            "prepare_fp8_layer_for_marlin": lambda layer, size_k_first: None,
        },
        "vllm.utils": {},
        "vllm.utils.deep_gemm": {
            "fp8_gemm_nt": fp8_gemm_nt,
            "is_deep_gemm_e8m0_used": lambda: False,
            "is_deep_gemm_supported": lambda: True,
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
    return made, given


def test_bench_marlin_runs_the_engines_kernels_and_says_when_deepgemm_needs_cuda_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _module("flatshare_a100_bench_marlin", A100 / "bench_marlin.py")
    cuda = _Cuda()
    made, given = _fake_kernels(monkeypatch, cuda)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    rows = bench.run([1, 16], 5)
    cells = {(row.shape, row.m, row.kernel): row for row in rows}
    assert len(cells) == len(rows) == len(bench.SHAPES) * 2 * 3
    for shape, *_ in bench.SHAPES:
        for m in (1, 16):
            # Each kernel's GPU time a call, though Marlin's Python takes ten times as long to launch it.
            assert cells[shape, m, "marlin"].us == pytest.approx(40.0)
            assert cells[shape, m, "bf16"].us == pytest.approx(20.0)
            # DeepGEMM's failed build, said in one line: what to set, the nvcc it took and the compiler's error.
            row = cells[shape, m, "deepgemm"]
            assert row.us is None
            assert row.note.startswith("DeepGEMM needs nvcc: set CUDA_HOME"), row.note
            assert "/usr/local/cuda/bin/nvcc" in row.note and "\n" not in row.note
            assert 'error: #error "DeepGEMM SM120 kernels require CUDA 13.0 or newer"' in row.note
    # One lock workspace a layer, made as vLLM's engine makes its own (MARLIN_MAX_BLOCKS_PER_SM, 4, blocks an SM)
    # and given to every call: no call allocates and zeroes one (workspace=None).
    assert [blocks for _, blocks in made] == [4] * len(bench.SHAPES)
    assert set(given) == {f"workspace {i + 1}" for i in range(len(bench.SHAPES))}
    bench.write_csv(rows, tmp_path / "gemm.csv")
    lines = (tmp_path / "gemm.csv").read_text().splitlines()
    assert lines[0] == "shape,n,k,m,kernel,us,tflops,timing,note"
    assert len(lines) == 1 + len(rows)  # a line a row
    assert lines[1] == "qkv,6144,2560,1,marlin,40.00,0.8,cuda-graph,"
    # With the engine unit's CUDA_HOME, DeepGEMM builds: each call quantizes the activations and multiplies.
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-13.0")
    deepgemm = [row for row in bench.run([1], 5) if row.kernel == "deepgemm"]
    assert [row.us for row in deepgemm] == [pytest.approx(28.0)] * len(bench.SHAPES)


ENGINE_PYTHON = """#!/usr/bin/env bash
# The box's engine venv Python in the session-2 test: host.py runs on the test's Python; bench_marlin.py logs its
# arguments and CUDA_HOME and writes the header of the CSV the graph-timed bench writes.
case ${1##*/} in
  host.py) exec "$FAKE_PYTHON" "$@" ;;
  bench_marlin.py) ;;
  *) exit 1 ;;
esac
echo "bench_marlin.py ${*:2} CUDA_HOME=${CUDA_HOME:-}" >>"$HOME/bench.log"
while (($#)); do
  [[ $1 == --out ]] && out=$2
  shift
done
echo shape,n,k,m,kernel,us,tflops,timing,note >"$out"
"""


def test_session_2_times_the_gemms_in_the_engine_units_environment_and_not_by_an_older_csv(
    tmp_path: Path,
) -> None:
    hosts, out = tmp_path / "hosts", tmp_path / "out"
    box = hosts / "gpu"
    runs = box / "flatshare" / "runs" / "marlin-dry"
    runs.mkdir(parents=True)
    # The first bench's gemm.csv, which timed one call between two events.
    first = "shape,n,k,m,kernel,us,tflops,note\nqkv,6144,2560,1,marlin,183.50,0.2,\n"
    (runs / "gemm.csv").write_text(first)
    src = box / "flatshare" / "src"
    for rel in (
        "deploy/a100/host.py",
        "deploy/phastform/systemd/flatshare-engine.service",
        "benchmarks/a100/bench_marlin.py",
    ):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).symlink_to(REPO / rel)
    # The engine variant of step 2 fails, which ends the session after its GEMMs.
    (src / "deploy" / "phastform" / "variant.sh").write_text(
        'echo "variant.sh $*" >>"$HOME/variant.log"; exit 1\n'
    )
    python = box / "flatshare" / "venvs" / "engine" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(ENGINE_PYTHON)
    python.chmod(0o755)
    env = os.environ | {
        "PATH": os.pathsep.join([str(FAKES / "campaign"), str(FAKES / "a100"), os.environ["PATH"]]),
        "FAKE_HOSTS": str(hosts),
        "FAKE_PYTHON": sys.executable,
        "FLATSHARE_GPU": "gpu",
        "FLATSHARE_OUT": str(out),
    }
    env.pop("CUDA_HOME", None)

    def session() -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(A100 / "marlin-session2.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    done = session()
    assert done.returncode == 1 and (box / "variant.log").exists(), (done.stdout, done.stderr)
    # The bench ran once, the engine stopped, with the CUDA_HOME of the box's engine unit; the first bench's CSV kept.
    ran = "bench_marlin.py --m 1,16,64,128,256,512,1024 --out flatshare/runs/marlin-dry/gemm.csv"
    assert (box / "bench.log").read_text() == f"{ran} CUDA_HOME=/usr/local/cuda-13.0\n"
    assert "systemctl --user stop flatshare-engine" in (box / "systemctl.log").read_text()
    assert (runs / "gemm-events.csv").read_text() == first
    assert (runs / "gemm.csv").read_text() == "shape,n,k,m,kernel,us,tflops,timing,note\n"
    # The next session finds the graph-timed CSV and does not run the bench again.
    done = session()
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert (box / "bench.log").read_text().count("bench_marlin.py") == 1
    assert "gemm.csv: done before" in (out / "session.log").read_text()
