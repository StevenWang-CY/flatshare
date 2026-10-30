"""How E2's runs wait for the tools and services they start: polling, every POLL_S."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from flatshare_bench.e2.arm import E2Error

POLL_S = 1.0
"""How often a run checks on the tools and services it waits for."""


def until(done: Callable[[], bool], timeout: float, message: str) -> None:
    """Return once ``done()``, asked every POLL_S; past ``timeout`` seconds, raise an E2Error that
    says ``message``."""
    deadline = time.monotonic() + timeout
    while not done():
        if time.monotonic() > deadline:
            raise E2Error(message)
        time.sleep(POLL_S)


def pause(at_most: float = math.inf) -> None:
    """Wait before a run looks again at what it waits for: POLL_S, or ``at_most`` seconds if that is
    less."""
    time.sleep(min(POLL_S, at_most))
