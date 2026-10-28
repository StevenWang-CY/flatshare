"""``flatshare-voice arrivals``: the options and the run of a soak's callers (:mod:`.arrivals`), which the command line
(:mod:`.cli`) adds to its commands; they live here, beside the callers, rather than in the shared command line."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
from pathlib import Path

from . import cli


def add_arrivals(commands: argparse._SubParsersAction) -> None:
    """Add ``arrivals`` to the command line's commands."""
    arrivals = commands.add_parser(
        "arrivals",
        help="call the voice agent on a schedule for hours, recording every attempt (a soak's callers)",
        description="Calls that arrive one at a time, a Poisson process of --calls-per-hour, each of --turns turns "
        "with the callers' think times, through LiveKit inside the voice-plane namespace, for --duration seconds. "
        "Every attempt goes to attempts.jsonl: accepted, refused (no agent came; why, from the voice worker's "
        "status file), failed (no room joined) or skipped (--max-open attempts already open); every turn to "
        "turns.jsonl as bots writes it, with when it ended. SIGINT or SIGTERM stops new calls and ends each call "
        "with its current turn; a second one hangs them up. LIVEKIT_API_KEY and LIVEKIT_API_SECRET come from the "
        "environment.",
    )
    arrivals.add_argument(
        "--url",
        default=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:17880"),
        help="LiveKit signalling URL (env LIVEKIT_URL; default %(default)s)",
    )
    arrivals.add_argument(
        "--corpus", type=Path, required=True, help="directory from render-callers"
    )
    arrivals.add_argument(
        "--calls-per-hour", type=_above_zero, required=True, help="the calls' mean arrival rate"
    )
    arrivals.add_argument(
        "--duration", type=_seconds, required=True, help="seconds over which calls arrive"
    )
    arrivals.add_argument(
        "--turns",
        type=cli._positive,
        default=10,
        help="measured turns per call (default %(default)s)",
    )
    arrivals.add_argument(
        "--barge-in",
        type=cli._share,
        default=0.25,
        help="share of turns with an interruption, from 0 to 1 (default %(default)s)",
    )
    arrivals.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed of the arrival times and of each call's lines, interruptions and pauses "
        "(default %(default)s)",
    )
    arrivals.add_argument(
        "--warmup-turns",
        type=cli._count,
        default=1,
        help="unmeasured turns that open each call, as bots' (default %(default)s: a serving worker's AEC "
        "warm-up ignores an interruption of its first reply)",
    )
    arrivals.add_argument(
        "--run",
        type=cli._count,
        default=0,
        help="the run's number in its soak (default %(default)s)",
    )
    arrivals.add_argument(
        "--first-call",
        type=cli._count,
        default=0,
        help="the number of the run's first call: the soak's calls so far (default %(default)s)",
    )
    arrivals.add_argument(
        "--dispatch-timeout",
        type=_above_zero,
        default=15.0,
        help="seconds a caller waits for the agent before the call counts as refused (default %(default)s)",
    )
    arrivals.add_argument(
        "--max-open",
        type=cli._positive,
        default=8,
        help="attempts open at once at most; a call due beyond them is skipped (default %(default)s)",
    )
    arrivals.add_argument(
        "--worker-status",
        type=Path,
        help="the voice worker's status file, which says why a call was refused (default "
        "$FLATSHARE_VOICE_STATUS, else $XDG_RUNTIME_DIR/flatshare/voice-worker.json)",
    )
    arrivals.add_argument("--out", type=Path, required=True, help="run directory")
    arrivals.set_defaults(func=_arrivals)


def _arrivals(args: argparse.Namespace) -> int:
    from .arrivals import ArrivalsConfig, default_status_file, run_arrivals
    from .bots import BotsConfig

    api_key, api_secret = (os.environ[name] for name in cli.LIVEKIT_CREDENTIALS)
    bots = BotsConfig(
        url=args.url,
        api_key=api_key,
        api_secret=api_secret,
        corpus=args.corpus,
        callers=1,
        turns=args.turns,
        barge_in=args.barge_in,
        seed=args.seed,
        warmup_turns=args.warmup_turns,
    )
    config = ArrivalsConfig(
        bots=bots,
        calls_per_hour=args.calls_per_hour,
        duration_s=args.duration,
        run=args.run,
        first_call=args.first_call,
        dispatch_timeout=args.dispatch_timeout,
        max_open=args.max_open,
        worker_status=args.worker_status or default_status_file(),
    )
    summary = asyncio.run(run_arrivals(config, args.out))
    gc.collect()  # the LiveKit handles go while the native library is loaded, as bots' do
    print(json.dumps(summary, indent=1))
    return 0


def _above_zero(text: str) -> float:
    """A number above 0, for argparse."""
    value = float(text)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError(f"must be above 0, got {value}")
    return value


def _seconds(text: str) -> float:
    """Seconds, 0 or more, for argparse."""
    value = float(text)
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError(f"must be 0 or more seconds, got {value}")
    return value
