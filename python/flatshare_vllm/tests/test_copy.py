"""flatshare_copy (src/flatshare_vllm/serve/flatshare_copy.py) on CPU: it takes vLLM's own copy selection with only the
size threshold lifted, so the box's 64 KiB pages, which vLLM sends to the copy engine, go to the SM kernel; it refuses
where vLLM's other conditions fail; and the serve module installs it only when asked and records what it installed. The
kernel itself runs on the card only (the install checks it there before the first load)."""

import functools
import importlib
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.kv_offload.cpu.gpu_worker")

from test_make_room import module, scheduler  # noqa: F401
from test_queue import SERVE, SWITCHES, serve  # noqa: F401
from vllm import _custom_ops as ops
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.v1.kv_offload.cpu import gpu_worker, swap_blocks_triton

PAGE = 64 * 1024
"""The box's page, as the install reported it there: 16 tokens of one layer's K and V, 8 KV heads of 128 in bf16."""


@pytest.fixture
def flatshare_copy(monkeypatch):
    monkeypatch.syspath_prepend(str(SERVE))
    monkeypatch.delitem(sys.modules, "flatshare_copy", raising=False)
    copy = importlib.import_module("flatshare_copy")
    # The CPU test box has no usable Triton: stand in for what vLLM's selection asks of it, on a CUDA platform.
    monkeypatch.setattr(gpu_worker, "HAS_TRITON", True)
    monkeypatch.setattr(
        gpu_worker, "triton", SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    )
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", True)
    monkeypatch.setattr(swap_blocks_triton, "NUM_SMS", swap_blocks_triton.NUM_SMS)
    warmed = []
    monkeypatch.setattr(copy, "warm", lambda fn, page: warmed.append(page))
    copy.warmed = warmed
    return copy


def connector(monkeypatch, pinned=True, pages=(PAGE,)):
    handler = SimpleNamespace(
        layer_refs_per_group=[[SimpleNamespace(page_size_bytes=p) for p in pages]],
        _swap_blocks_batch=ops.swap_blocks_batch,
    )
    region = SimpleNamespace(is_pinned=pinned)
    worker = SimpleNamespace(_load_handler=handler, _mmap_region=region)
    monkeypatch.setattr(
        kv_transfer_state,
        "_KV_CONNECTOR_AGENT",
        SimpleNamespace(connector_worker=SimpleNamespace(worker=worker)),
    )
    return handler


def test_vllm_sends_the_boxs_pages_to_the_copy_engine(flatshare_copy, monkeypatch):
    """The negative control: without the install, vLLM's selection for these pages is the copy engine, and for pages
    under its threshold the SM kernel."""
    handler = connector(monkeypatch)
    assert (
        gpu_worker._select_swap_blocks_fn(handler.layer_refs_per_group, False, True)
        is ops.swap_blocks_batch
    )
    assert gpu_worker._select_swap_blocks_fn(
        [[SimpleNamespace(page_size_bytes=16 * 1024)]], False, True
    ).func is (swap_blocks_triton.swap_blocks_batch)


def test_loads_go_to_the_sm_kernel(flatshare_copy, monkeypatch):
    handler = connector(monkeypatch)
    report = flatshare_copy.install(4)
    assert isinstance(handler._swap_blocks_batch, functools.partial)
    assert handler._swap_blocks_batch.func is swap_blocks_triton.swap_blocks_batch
    assert handler._swap_blocks_batch.keywords == {"bytes_per_chunk": 8192}
    assert swap_blocks_triton.NUM_SMS == 4
    assert gpu_worker.THRESHOLD_BYTES == swap_blocks_triton.THRESHOLD_BYTES == 28 * 1024
    assert flatshare_copy.warmed == [PAGE]
    assert report == {"sm_load_programs": 4, "bytes_per_chunk": 8192, "page_bytes": [PAGE]}


@pytest.mark.parametrize("case", ["no connector", "unpinned", "odd pages"])
def test_it_refuses_where_vllms_conditions_fail(flatshare_copy, monkeypatch, case):
    if case == "no connector":
        monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", None)
        handler = None
    else:
        handler = connector(
            monkeypatch,
            pinned=case != "unpinned",
            pages=(PAGE, 36) if case == "odd pages" else (PAGE,),
        )
    with pytest.raises(RuntimeError, match="flatshare_copy"):
        flatshare_copy.install(4)
    if handler is not None:
        assert handler._swap_blocks_batch is ops.swap_blocks_batch
    assert gpu_worker.THRESHOLD_BYTES == 28 * 1024
    assert flatshare_copy.warmed == []


@pytest.mark.parametrize(
    "table, want",
    [({"sm_load_programs": 4}, 4), ({"sm_load_programs": 0}, 0), (None, 0)],
)
def test_the_serve_module_installs_it_only_when_asked(serve, model, monkeypatch, table, want):  # noqa: F811
    calls = []
    monkeypatch.setattr(
        serve.flatshare_copy,
        "install",
        lambda programs: (
            calls.append(programs) or {"sm_load_programs": programs, "page_bytes": [PAGE]}
        ),
    )
    additional = {"flatshare_copy": table} if table is not None else None
    sched = scheduler(
        serve.AsyncScheduler, model, 512, dict(SWITCHES), max_model_len=4096, additional=additional
    )
    assert calls == ([want] if want else [])
    assert serve.effective(sched)["copy"]["sm_load_programs"] == want


def test_a_negative_count_is_refused(serve, model):  # noqa: F811
    with pytest.raises(ValueError, match="sm_load_programs"):
        scheduler(
            serve.AsyncScheduler,
            model,
            512,
            dict(SWITCHES),
            max_model_len=4096,
            additional={"flatshare_copy": {"sm_load_programs": -1}},
        )


def test_the_effective_file_records_the_copy_installed(serve, model, tmp_path, monkeypatch):  # noqa: F811
    """The file serving-check reads is written once the copy is installed, so it says what runs."""
    import json

    path = tmp_path / "engine-effective.json"
    monkeypatch.setenv("FLATSHARE_ENGINE_EFFECTIVE", str(path))
    monkeypatch.setattr(
        serve.flatshare_copy,
        "install",
        lambda programs: {"sm_load_programs": programs, "page_bytes": [PAGE]},
    )
    scheduler(
        serve.AsyncScheduler,
        model,
        512,
        dict(SWITCHES),
        max_model_len=4096,
        additional={"flatshare_copy": {"sm_load_programs": 4}},
    )
    assert json.loads(path.read_text())["copy"] == {"sm_load_programs": 4, "page_bytes": [PAGE]}
