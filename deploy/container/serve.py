#!/usr/bin/env python3
"""The serving engine in a container: ``vllm serve`` with make-room's scheduler at the serving profile's settings, its
start then held to that profile, as the box's unit holds it (deploy/phastform/variants/make-room-serve).

    serve.py [VLLM_FLAG...]

The engine's flags come from serving.json, the effective configuration that deploy/profile.py renders from the
serving profile (deploy/profiles/rtx-5060-ti-serve.toml): the step cap and the long-prefill threshold over
engine.yaml's, the scheduler class, and make-room's switches. Flags given to the container follow them, such as
``--uds /run/flatshare/sock/engine.sock`` or ``--host 0.0.0.0 --port 8000``. Once the engine has written the configuration
it runs with (``FLATSHARE_ENGINE_EFFECTIVE``), deploy/bin/serving-check holds it to serving.json: an engine that runs
anything else is stopped, and the container exits 1, as the unit's start fails. SIGTERM and SIGINT reach the engine
as SIGINT, the unit's stop. Standard library only, beside the image's own Python.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.environ.get("FLATSHARE_HOME", "/opt/flatshare"))
"""Where the image keeps the repository's files the engine needs, at their own paths, as a release on the box does."""
SERVING = HOME / "deploy" / "phastform" / "serving.json"
ENGINE_CONFIG = HOME / "deploy" / "phastform" / "engine.yaml"
SERVING_CHECK = HOME / "deploy" / "bin" / "serving-check"


def flags(serving: dict, config: Path = ENGINE_CONFIG) -> list[str]:
    """The flags of ``vllm serve`` that the serving profile sets, in the make-room-serve drop-in's order."""
    engine = serving["engine"]
    scheduler = engine["scheduler"]
    args = [
        "--config",
        str(config),
        "--max-num-batched-tokens",
        str(scheduler["max_num_batched_tokens"]),
        "--long-prefill-token-threshold",
        str(scheduler["long_prefill_token_threshold"]),
        "--scheduler-cls",
        engine["scheduler_cls"],
    ]
    for key, value in engine["make_room"].items():
        args += [f"--additional-config.make_room.{key}", json.dumps(value)]
    return args


def main(argv: list[str]) -> int:
    serving = json.loads(SERVING.read_text())
    effective = Path(
        os.environ.setdefault("FLATSHARE_ENGINE_EFFECTIVE", "/run/flatshare/engine-effective.json")
    )
    effective.parent.mkdir(parents=True, exist_ok=True)
    effective.unlink(missing_ok=True)
    engine = subprocess.Popen(["vllm", "serve", *flags(serving), *argv])

    def stop(signum: int, frame: object) -> None:
        engine.send_signal(signal.SIGINT)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # The serve module writes the configuration whole (a file renamed into place) from the engine's core process, a
    # child of this one, as the scheduler starts.
    while engine.poll() is None and not effective.exists():
        time.sleep(1)
    if engine.poll() is not None:
        return engine.returncode
    check = [
        sys.executable,
        str(SERVING_CHECK),
        str(SERVING),
        "--engine",
        str(effective),
        "--pid",
        str(engine.pid),
    ]
    if subprocess.run(check).returncode != 0:
        print(
            "serve.py: the engine does not run the serving profile; stopping it",
            file=sys.stderr,
            flush=True,
        )
        engine.send_signal(signal.SIGINT)
        engine.wait()
        return 1
    return engine.wait()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
