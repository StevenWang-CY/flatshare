"""The ``flatshare-voice`` command line: render the caller corpus and run measurement callers.

The voice agent itself is ``flatshare-voice-worker`` (LiveKit's worker CLI, ``flatshare-voice-worker start``).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
from pathlib import Path


def _render(args: argparse.Namespace) -> int:
    from .callers import render

    manifest = render(args.tts_url, args.out, model=args.model, uds=args.uds)
    print(f"wrote {len(manifest)} clips to {args.out}")
    return 0


LIVEKIT_CREDENTIALS = ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
"""The LiveKit server's key and secret, which ``bots`` signs its callers' join tokens with, from the environment."""


def _bots(args: argparse.Namespace) -> int:
    from .bots import BotsConfig, run_bots

    api_key, api_secret = (os.environ[name] for name in LIVEKIT_CREDENTIALS)
    config = BotsConfig(
        url=args.url,
        api_key=api_key,
        api_secret=api_secret,
        corpus=args.corpus,
        callers=args.callers,
        turns=args.turns,
        barge_in=args.barge_in,
        seed=args.seed,
        warmup_turns=args.warmup_turns,
        in_order=args.in_order,
        save_audio=args.save_audio,
    )
    summary = asyncio.run(run_bots(config, args.out))
    # Release the LiveKit handles while the native library is still loaded; left to interpreter
    # shutdown, each one prints a failed assertion.
    gc.collect()
    print(json.dumps(summary, indent=1))
    return 0 if not summary["errors"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flatshare-voice", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser(
        "render-callers",
        help="render the scripted caller lines to 16 kHz WAVs with a TTS server",
        description="Render the caller corpus once through an OpenAI-compatible TTS server: "
        "16 kHz mono WAVs and manifest.json.",
    )
    render.add_argument("out", type=Path, help="corpus directory")
    render.add_argument(
        "--tts-url", default="http://127.0.0.1:8880/v1", help="TTS base URL (default %(default)s)"
    )
    render.add_argument("--uds", help="reach the TTS server through this Unix socket")
    render.add_argument("--model", default="kokoro", help="TTS model (default %(default)s)")
    render.set_defaults(func=_render)

    bots = commands.add_parser(
        "bots",
        help="call the voice agent and measure voice-to-voice latency and barge-in",
        description="Run callers against the voice agent through LiveKit (inside the voice-plane "
        "namespace). LIVEKIT_API_KEY and LIVEKIT_API_SECRET come from the environment.",
    )
    bots.add_argument(
        "--url",
        default=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:17880"),
        help="LiveKit signalling URL (env LIVEKIT_URL; default %(default)s)",
    )
    bots.add_argument("--corpus", type=Path, required=True, help="directory from render-callers")
    bots.add_argument(
        "--callers", type=_positive, default=2, help="concurrent calls (default %(default)s)"
    )
    bots.add_argument(
        "--turns", type=_positive, default=20, help="turns per call (default %(default)s)"
    )
    bots.add_argument(
        "--barge-in",
        type=_share,
        default=0.25,
        help="share of turns with an interruption, from 0 to 1 (default %(default)s)",
    )
    bots.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed of the lines, interruptions and pauses (default %(default)s)",
    )
    bots.add_argument(
        "--warmup-turns",
        type=_count,
        default=0,
        help="turns that open each call unmeasured: a line and its reply, no interruption, "
        'marked "warmup" in turns.jsonl and left out of the summary (default %(default)s)',
    )
    bots.add_argument(
        "--in-order",
        action="store_true",
        help="play the corpus's turn lines in order rather than drawing each: caller i from line "
        "i x TURNS on, every line once when TURNS is the number of lines and there is one caller",
    )
    bots.add_argument(
        "--save-audio",
        action="store_true",
        help="also write each call's audio and its turns' times, once the calls have ended "
        "(call-<i>.wav, call-<i>.json)",
    )
    bots.add_argument("--out", type=Path, required=True, help="run directory")
    bots.set_defaults(func=_bots)
    from .arrivals_command import add_arrivals  # a soak's callers, their options beside them

    add_arrivals(commands)
    return parser


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ("bots", "arrivals") and (
        missing := [name for name in LIVEKIT_CREDENTIALS if not os.environ.get(name)]
    ):
        parser.error(
            f"{args.command} needs {' and '.join(missing)} in the environment for its join tokens"
        )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
