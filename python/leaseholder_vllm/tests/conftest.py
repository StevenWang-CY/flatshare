"""The model the CPU suites build their schedulers for (tests/test_make_room.py, tests/test_trace.py), and
LH_NO_SKIPS, which makes a skipped test fail the session."""

import json
import os

import pytest


@pytest.fixture(scope="session")
def model(tmp_path_factory) -> str:
    """A model directory with only a config: the scheduler reads the model's configuration, never its weights."""
    path = tmp_path_factory.mktemp("model")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 1,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "torch_dtype": "float16",
    }
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """With LH_NO_SKIPS set, as CI's make-room job sets it, any skipped test fails the session. The CPU suites skip
    when their environment lacks torch or vLLM's scheduler, and a suite that skips checks nothing."""
    if not os.environ.get("LH_NO_SKIPS"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped and session.exitstatus == pytest.ExitCode.OK:
        reporter.write("\n")
        reporter.write_line(f"LH_NO_SKIPS: {len(skipped)} skipped, so the session fails", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
