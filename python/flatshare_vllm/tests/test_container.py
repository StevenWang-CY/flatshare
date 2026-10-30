"""The container recipe (deploy/container) against the box's serving variant: the entrypoint gives `vllm serve` the
make-room-serve drop-in's own flags, read from serving.json, and engine.yaml from the same path in the tree; the image
is vLLM's at the serving profile's version; and the build context (.dockerignore) lets in every file the image and the
entrypoint use, at its path in the repository, as serving-check expects, and nothing else."""

import importlib.machinery
import importlib.util
import json
import re
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTAINER = REPO / "deploy" / "container"
SERVING = REPO / "deploy" / "phastform" / "serving.json"
DROP_IN = REPO / "deploy" / "phastform" / "variants" / "make-room-serve" / "flatshare-engine.conf"


def load(name: str, path: Path):
    """The Python file at `path` as a module, whatever its name ends in (serving-check has no suffix)."""
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(module)
    return module


def entrypoint():
    return load("flatshare_container_serve", CONTAINER / "serve.py")


def drop_in_flags(text: str) -> list[str]:
    """The flags the drop-in's ExecStart gives `vllm serve`, before its socket and the unit's free flags."""
    line = next(
        line for line in text.splitlines() if line.startswith("ExecStart=") and " serve " in line
    )
    words = shlex.split(line.removeprefix("ExecStart="))
    return words[words.index("serve") + 1 : words.index("--uds")]


def test_the_entrypoint_gives_vllm_the_drop_ins_flags() -> None:
    serve = entrypoint()
    flags = serve.flags(json.loads(SERVING.read_text()))
    expected = drop_in_flags(DROP_IN.read_text())
    # engine.yaml at its path in the tree: the box's release under %h/flatshare/src, the image's under /opt/flatshare.
    assert expected[0] == flags[0] == "--config"
    assert expected[1] == "%h/flatshare/src/" + str(serve.ENGINE_CONFIG.relative_to(serve.HOME))
    assert flags[1] == str(serve.ENGINE_CONFIG) and flags[2:] == expected[2:]
    assert "--scheduler-cls" in flags and "--additional-config.make_room.budget_cap_tokens" in flags


def test_a_drop_in_with_another_value_differs() -> None:
    changed = DROP_IN.read_text().replace("budget_cap_tokens 48", "budget_cap_tokens 16")
    flags = entrypoint().flags(json.loads(SERVING.read_text()))
    assert flags[2:] != drop_in_flags(changed)[2:]


def test_the_image_is_vllms_at_the_profiles_version() -> None:
    version = json.loads(SERVING.read_text())["engine"]["vllm_version"]
    dockerfile = (CONTAINER / "Dockerfile").read_text()
    assert f"ARG VLLM_IMAGE=docker.io/vllm/vllm-openai:v{version}\n" in dockerfile


def test_every_compile_cache_is_under_the_volumes_directory() -> None:
    # The caches the A100 profile keeps across starts under cache/, and the CUDA driver's, which a host keeps in ~/.nv,
    # each under the directory the README's volume keeps.
    profile = (REPO / "deploy" / "profiles" / "a100-40gb.toml").read_text()
    kept = set(re.findall(r"Environment=(\w+)=%h/flatshare/cache/", profile))
    assert kept == {
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
    }
    image = dict(
        re.findall(r"\b(\w+)=(/opt/flatshare/cache/\S+)", (CONTAINER / "Dockerfile").read_text())
    )
    assert set(image) == kept | {"CUDA_CACHE_PATH"} and len(set(image.values())) == len(image)
    assert ":/opt/flatshare/cache " in (CONTAINER / "README.md").read_text()


def test_the_build_context_lets_in_every_file_used_and_nothing_else() -> None:
    serve = entrypoint()
    dockerfile = (CONTAINER / "Dockerfile").read_text()
    assert "COPY . /opt/flatshare/\n" in dockerfile and str(serve.HOME) == "/opt/flatshare"
    # What the image and the entrypoint use, by path in the tree; make-room's module is the build's own, and the
    # caches are the volume's.
    named = {
        path
        for path in re.findall(r"/opt/flatshare/([\w./-]+)", dockerfile)
        if path != "make-room" and not path.startswith("cache/")
    }
    named |= {
        str(path.relative_to(serve.HOME))
        for path in (serve.SERVING, serve.ENGINE_CONFIG, serve.SERVING_CHECK)
    }
    check = load("flatshare_serving_check", REPO / "deploy" / "bin" / "serving-check")
    named.add(str(check.PATCHES))
    ignore = (REPO / ".dockerignore").read_text().splitlines()
    assert ignore[ignore.index("*") + 1 :][0].startswith("!")
    allowed = [line[1:].rstrip("/") for line in ignore if line.startswith("!")]
    for path in named:
        assert (REPO / path).exists(), path
        assert any(path == entry or path.startswith(entry + "/") for entry in allowed), path
    for entry in allowed:
        assert any(path == entry or path.startswith(entry + "/") for path in named), (
            f"{entry} is let in unused"
        )
