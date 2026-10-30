"""What the voice command line's commands share: the LiveKit credentials' names and argparse's types for counts
and shares."""

from __future__ import annotations

import argparse

LIVEKIT_CREDENTIALS = ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
"""The LiveKit server's key and secret, which ``bots`` signs its callers' join tokens with, from the environment."""


def _count(text: str) -> int:
    """A count of 0 or more, for argparse."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _positive(text: str) -> int:
    """A count of 1 or more, for argparse."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def _share(text: str) -> float:
    """A share from 0 to 1, for argparse."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be a share from 0 to 1, got {value}")
    return value
