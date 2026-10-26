"""The ``lh-bench`` command line: trace scaling, latency probes, engine metrics and E2 points."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import signal
import sys
from pathlib import Path

from leaseholder_bench.load import LoadSpec, parse_lengths
from leaseholder_bench.protocol import DEFAULT_MODEL, SloClass
from leaseholder_bench.transport import runtime_socket

ADDRESS = "http://host:port, or unix: and a socket's path"

Commands = argparse._SubParsersAction  # what add_subparsers returns


def _scale(args: argparse.Namespace) -> int:
    from leaseholder_bench.traces import scale_corpus

    ids = None
    if args.ids is not None:
        ids = {line.strip() for line in args.ids.read_text().splitlines() if line.strip()}
    summary = scale_corpus(args.src, args.out_dir, args.factor, ids=ids, limit=args.limit)
    print(
        f"wrote {summary.traces} traces ({summary.requests} requests) to {args.out_dir}; "
        f"largest in+out is {summary.peak_tokens} tokens"
    )
    if ids is not None and summary.traces < len(ids):
        print(
            f"warning: {len(ids) - summary.traces} listed trace ids were not found", file=sys.stderr
        )
    return 0


def _add_scale_traces(commands: Commands) -> None:
    scale = commands.add_parser(
        "scale-traces",
        help="shrink Weka traces by a factor of tokens per hash block",
        description="Shrink every request of a Weka trace corpus by the same factor, keeping "
        "hash ids, sessions, subagents and think times. Writes one <trace id>.json per trace.",
    )
    scale.add_argument("src", type=Path, help="Weka traces: a .jsonl file or a directory of *.json")
    scale.add_argument("out_dir", type=Path, help="directory for the scaled traces")
    scale.add_argument(
        "--factor",
        type=int,
        default=32,
        help="tokens-per-block shrink factor: 32 fits a 32k context, 64 a 16k one "
        "(default %(default)s)",
    )
    scale.add_argument("--ids", type=Path, help="keep only the trace ids listed in this file")
    scale.add_argument("--limit", type=int, help="stop after this many traces")
    scale.set_defaults(func=_scale)


def _mock(args: argparse.Namespace) -> int:
    from leaseholder_bench.mock_server import serve

    options = {
        "model": args.model,
        "ttft_ms": args.ttft_ms,
        "itl_ms": args.itl_ms,
        "max_tokens": args.max_tokens,
    }
    if args.log is None:
        serve(args.port, None, **options)
        return 0
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("ab") as log:
        serve(args.port, log, **options)
    return 0


def _add_mock_server(commands: Commands) -> None:
    mock = commands.add_parser(
        "mock-server",
        help="run a recording OpenAI-compatible mock on 127.0.0.1",
        description="Streaming chat-completions mock bound to 127.0.0.1 that logs every "
        "request's headers and body fields.",
    )
    mock.add_argument(
        "--port", type=int, default=8000, help="port on 127.0.0.1 (default %(default)s)"
    )
    mock.add_argument("--log", type=Path, help="append one JSON line per request to this file")
    mock.add_argument(
        "--model", default="mock", help="model id listed by /v1/models (default %(default)s)"
    )
    mock.add_argument(
        "--ttft-ms",
        type=float,
        default=20.0,
        help="delay before the first token (default %(default)s)",
    )
    mock.add_argument(
        "--itl-ms", type=float, default=2.0, help="delay between tokens (default %(default)s)"
    )
    mock.add_argument(
        "--max-tokens", type=int, default=64, help="cap on streamed tokens (default %(default)s)"
    )
    mock.set_defaults(func=_mock)


def _kv_metrics(args: argparse.Namespace) -> int:
    from leaseholder_bench.kv_metrics import compare, fetch

    if args.action == "snapshot":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(fetch(args.address))
        return 0
    print(compare(args.before.read_text(), args.after.read_text()))
    return 0


def _add_kv_metrics(commands: Commands) -> None:
    kv = commands.add_parser(
        "kv-metrics",
        help="snapshot vLLM's /metrics, or compare KV block lifecycle histograms across a run",
        description="Snapshot an engine's /metrics before and after a run, then compare vLLM's "
        "KV block reuse-gap, idle-before-evict and lifetime histograms over that interval "
        "(the engine needs --kv-cache-metrics).",
    )
    actions = kv.add_subparsers(dest="action", required=True)
    snap = actions.add_parser("snapshot", help="save a server's /metrics to a file")
    snap.add_argument(
        "address",
        help=f"the server ({ADDRESS}), e.g. {runtime_socket('phastform-engine.sock')}, the engine's tunnel",
    )
    snap.add_argument("out", type=Path)
    cmp_ = actions.add_parser("compare", help="report the histograms between two snapshots")
    cmp_.add_argument("before", type=Path)
    cmp_.add_argument("after", type=Path)
    kv.set_defaults(func=_kv_metrics)


def _tiers(args: argparse.Namespace) -> int:
    from leaseholder_bench.tiers import TiersConfig, render, run_tiers

    config = TiersConfig(
        url=args.url,
        model=args.model,
        lengths=args.lengths,
        reps=args.reps,
        background=args.background,
        decode_tokens=args.decode_tokens,
        priority=None if args.no_priority else args.priority,
        seed=args.seed,
    )
    summary = asyncio.run(run_tiers(config, args.out))
    print(render(summary))
    return 0 if not summary["invalid"] else 3


def _add_tiers(commands: Commands) -> None:
    tiers = commands.add_parser(
        "tiers",
        help="measure the time to first token from the GPU pool, the CPU tier and recompute (E0)",
        description="For each prompt length, send a unique prompt three times: cold (recomputed), "
        "again (GPU prefix-cache hit), and after flushing the GPU pool (restored from the CPU "
        "tier). vLLM's prefix-cache counters verify each request's source. Writes run.json, "
        "samples.jsonl and summary.json.",
    )
    tiers.add_argument(
        "--url",
        default=runtime_socket("phastform-engine.sock"),
        help=f"the engine ({ADDRESS}; default %(default)s, its tunnel on cyan); it needs /metrics and "
        "an offload tier",
    )
    tiers.add_argument("--model", default=DEFAULT_MODEL, help="model name (default %(default)s)")
    tiers.add_argument(
        "--lengths",
        type=parse_lengths,
        default="1024,2048,4096,8192,16384,24576,31744",
        help="comma-separated prompt lengths in tokens (default %(default)s)",
    )
    tiers.add_argument(
        "--reps", type=int, default=5, help="samples per length (default %(default)s)"
    )
    tiers.add_argument("--out", type=Path, required=True, help="run directory")
    tiers.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed of the prompts and the length order (default %(default)s)",
    )
    tiers.add_argument(
        "--priority",
        type=int,
        default=1,
        help="vLLM priority of the probe requests (default %(default)s); the load runs at 2",
    )
    _add_load_arguments(tiers)
    tiers.set_defaults(func=_tiers)


def _voice_turn(args: argparse.Namespace) -> int:
    from leaseholder_bench.voice_turn import VoiceTurnConfig, render, run_voice_turns

    config = VoiceTurnConfig(
        stt=args.stt,
        llm=args.llm,
        tts=args.tts,
        stt_model=args.stt_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        voice=args.voice,
        audio=args.audio,
        reps=args.reps,
        warm=args.warm,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        slo_class=args.slo_class,
        send_priority=not args.no_priority,
        background=args.background,
        decode_tokens=args.decode_tokens,
    )
    print(render(asyncio.run(run_voice_turns(config, args.out))))
    return 0


def _add_voice_turn(commands: Commands) -> None:
    turn = commands.add_parser(
        "voice-turn",
        help="time server-side voice turns: STT, LLM first sentence, TTS first audio",
        description="Run voice turns against OpenAI-compatible STT, LLM and TTS servers and time "
        "each stage; with --background, while agent load keeps the LLM busy. Writes run.json, "
        "turns.jsonl and summary.json.",
    )
    for server, socket in (
        ("stt", "phastform-stt.sock"),
        ("llm", "phastform-engine.sock"),
        ("tts", "phastform-tts.sock"),
    ):
        turn.add_argument(
            f"--{server}",
            default=runtime_socket(socket),
            help=f"the {server.upper()} server ({ADDRESS}), its OpenAI API at /v1 (default %(default)s, "
            "its tunnel on cyan)",
        )
    turn.add_argument(
        "--stt-model",
        default="openai/whisper-large-v3-turbo",
        help="STT model (default %(default)s)",
    )
    turn.add_argument("--llm-model", default=DEFAULT_MODEL, help="LLM model (default %(default)s)")
    turn.add_argument("--tts-model", default="kokoro", help="TTS model (default %(default)s)")
    turn.add_argument("--voice", default="af_heart", help="TTS voice (default %(default)s)")
    turn.add_argument(
        "--audio", type=Path, action="append", required=True, help="utterance WAV (repeatable)"
    )
    turn.add_argument(
        "--reps", type=int, default=20, help="measured turns per caller (default %(default)s)"
    )
    turn.add_argument(
        "--warm",
        type=int,
        default=2,
        help="unmeasured warm-up turns per caller (default %(default)s)",
    )
    turn.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="callers taking turns at once (default %(default)s)",
    )
    turn.add_argument(
        "--max-tokens", type=int, default=60, help="cap on each reply (default %(default)s)"
    )
    turn.add_argument(
        "--class",
        dest="slo_class",
        type=SloClass,
        choices=list(SloClass),
        default="realtime",
        help="SLO class of the turns: their X-LH-Class and vLLM priority (default %(default)s)",
    )
    turn.add_argument("--out", type=Path, required=True, help="run directory")
    _add_load_arguments(turn)
    turn.set_defaults(func=_voice_turn)


def _drill(args: argparse.Namespace) -> int:
    import orjson

    from leaseholder_bench.drills import (
        DrillConfig,
        agents_key,
        clients,
        default_uds,
        fill_cpu_tier,
        report,
        run_streams,
        start_phases,
    )
    from leaseholder_bench.e2.arm import E2Error
    from leaseholder_bench.e2.run import Host, exclusive
    from leaseholder_bench.transport import async_client

    if args.action == "streams":
        config = DrillConfig(
            realtime=args.realtime,
            background=args.background,
            max_tokens=args.max_tokens,
            deadline_s=args.deadline,
            probe_interval_s=args.probe_interval,
        )
        drill_clients = clients(config, args.gateway, args.uds or default_uds())
        engine = async_client(args.engine, timeout=10.0)
        host = Host.load()
        try:
            with exclusive(host):  # never beside an E2 point
                key = agents_key(host)
                asyncio.run(
                    run_streams(config, drill_clients, key, args.out, args.event_file, engine)
                )
        except E2Error as exc:
            print(f"lh-bench drill: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.action == "cpu-tier":
        try:
            with exclusive(Host.load()):  # nothing else may use the engine meanwhile
                engine = async_client(args.engine, timeout=300.0)
                record = asyncio.run(fill_cpu_tier(engine, args.out, factor=args.factor))
        except E2Error as exc:
            print(f"lh-bench drill: {exc}", file=sys.stderr)
            return 1
        print(orjson.dumps(record, option=orjson.OPT_INDENT_2).decode())
        return 0 if record["passed"] else 1
    if args.action == "start":
        for journal in args.journal:
            print(f"{journal.name}:")
            for phase, seconds in start_phases(journal.read_text()):
                print(f"  {seconds:7.1f} s  {phase}")
        return 0
    print(report(orjson.loads(args.record.read_bytes()), args.event, args.label))
    return 0


def _add_drill(commands: Commands) -> None:
    drill = commands.add_parser(
        "drill",
        help="the client side of a failure drill (ops/drills/drill.sh) and its summary",
        description="streams: open streaming completions through the gateway (realtime on the voice plane's "
        "socket, background on its TCP port with the agents tenant's key), print 'streaming' once each has its "
        "first byte, and read each to its end; from the moment EVENT_FILE holds the event's time, send a "
        "one-token completion on each plane every PROBE_INTERVAL seconds, each on a new connection, until every "
        "stream has ended and each plane's latest one succeeded; sample the engine's running and waiting "
        "requests meanwhile; writes the record to OUT. report: summarize a record. start: time a server's "
        "start from its journal.",
    )
    actions = drill.add_subparsers(dest="action", required=True)
    streams = actions.add_parser(
        "streams", help="open the streams, and probe from the event until requests succeed again"
    )
    streams.add_argument(
        "--gateway", default="http://127.0.0.1:18080", help="the gateway (default %(default)s)"
    )
    streams.add_argument(
        "--uds",
        type=Path,
        help="the gateway's voice-plane socket (default $XDG_RUNTIME_DIR/lh/gateway.sock)",
    )
    streams.add_argument(
        "--realtime", type=int, default=2, help="realtime streams (default %(default)s)"
    )
    streams.add_argument(
        "--background", type=int, default=4, help="background streams (default %(default)s)"
    )
    streams.add_argument(
        "--max-tokens", type=int, default=2000, help="each stream's (default %(default)s)"
    )
    streams.add_argument(
        "--deadline",
        type=float,
        default=600.0,
        help="seconds for the streams, then the probe (default %(default)s)",
    )
    streams.add_argument(
        "--probe-interval",
        type=float,
        default=1.0,
        help="seconds between probes (default %(default)s)",
    )
    streams.add_argument(
        "--event-file",
        type=Path,
        required=True,
        help="where the drill script writes the event's wall-clock time as the event happens",
    )
    streams.add_argument(
        "--engine",
        default=runtime_socket("phastform-engine.sock"),
        help=f"the engine, for its running requests ({ADDRESS}; default %(default)s, its tunnel on cyan)",
    )
    streams.add_argument("--out", type=Path, required=True, help="the record (JSON)")
    summary = actions.add_parser("report", help="summarize a drill's record")
    summary.add_argument("record", type=Path)
    summary.add_argument(
        "--event",
        type=float,
        help="the event's wall-clock time (default: the record's, as the client read it)",
    )
    summary.add_argument(
        "--label", default="the event", help="what --event names (default %(default)s)"
    )
    tier = actions.add_parser(
        "cpu-tier", help="fill the engine's CPU tier past its capacity and check what it then holds"
    )
    tier.add_argument(
        "--engine",
        default=runtime_socket("phastform-engine.sock"),
        help=f"the engine ({ADDRESS}; default %(default)s, its tunnel on cyan)",
    )
    tier.add_argument(
        "--factor", type=float, default=2.0, help="the tier's capacity stored (default %(default)s)"
    )
    tier.add_argument("--out", type=Path, required=True, help="the record (JSON)")
    start = actions.add_parser(
        "start", help="the phases of a server's start, from its journal (journalctl -o short-unix)"
    )
    start.add_argument("journal", type=Path, nargs="+")
    drill.set_defaults(func=_drill)


def _e2(args: argparse.Namespace) -> int:
    from leaseholder_bench.e2.arm import E2Error, load_arm
    from leaseholder_bench.e2.record import export_point, render
    from leaseholder_bench.e2.report import load_points, read_dots, write_report
    from leaseholder_bench.e2.run import Host, Systemd, gpu_sources, reset_cyan, run_point

    try:
        if args.action == "engine":
            print(load_arm(args.arm).engine)
        elif args.action == "reset":
            applied = reset_cyan(Host.load(), Systemd())
            for unit in ("gateway", "worker"):
                print(f"{unit}: {'restarted' if applied[unit]['restarted'] else 'unchanged'}")
        elif args.action == "run":
            # A closed ssh session does not stop a point; SIGINT and SIGTERM do, after cleaning up.
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, _interrupted)
            sources = gpu_sources(args.gpu_source) if args.gpu_source else None
            host = Host.load()
            if args.profile:
                host = dataclasses.replace(host, profile=args.profile)
            try:
                summary = run_point(
                    args.arm,
                    args.out,
                    host,
                    save_audio=args.save_audio,
                    gpu_source=sources,
                    agents_seed=args.agents_seed,
                    kv_sample_s=args.kv_sample_s,
                )
            except KeyboardInterrupt:
                print(f"lh-bench e2: {args.arm.stem} interrupted", file=sys.stderr)
                return 130
            print(render(summary))
        elif args.action == "export":
            records = []
            for run_dir in args.run_dirs:
                if (record := export_point(run_dir, args.out)) is None:
                    print(f"lh-bench e2: {run_dir}: the point did not complete", file=sys.stderr)
                else:
                    records.append(record)
                    print(record)
            if not records:
                print("lh-bench e2: no complete point to export", file=sys.stderr)
                return 1
        else:
            rows, incomplete = load_points(args.run_dirs)
            for run_dir in incomplete:
                print(f"lh-bench e2: {run_dir}: the point did not complete", file=sys.stderr)
            if not rows:
                print("lh-bench e2: no complete point to report", file=sys.stderr)
                return 1
            try:
                dots = read_dots(args.dots) if args.dots else ()
                written = write_report(rows, args.out, args.waterfall, dots)
            except ValueError as exc:  # a --waterfall point without turn lines, a bad --dots file
                print(f"lh-bench e2: {exc}", file=sys.stderr)
                return 1
            for path in written:
                print(path)
    except E2Error as exc:
        print(f"lh-bench e2: {exc}", file=sys.stderr)
        return 1
    return 0


def _interrupted(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)


def _seconds(text: str) -> float:
    """An interval in seconds, 0 or more."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds") from None
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r}: an interval is 0 or more seconds")
    return value


def _add_e2(commands: Commands) -> None:
    e2 = commands.add_parser(
        "e2",
        help="E2 points: measure one from its arm file on cyan, or build the frontier",
        description="Measure E2 points (voice-to-voice latency beside agent load) and place them on "
        "the frontier of background goodput against V2V p95.",
    )
    actions = e2.add_subparsers(dest="action", required=True)
    run = actions.add_parser(
        "run",
        help="measure one point on cyan",
        description="Configure cyan's units for the arm, check the engine, run the AIPerf replay and "
        "the calls, and write run.json, the tools' outputs, metric snapshots and summary.json.",
    )
    run.add_argument("arm", type=Path, help="arm file (TOML); the point is named after the file")
    run.add_argument("--out", type=Path, required=True, help="run directory")
    run.add_argument(
        "--save-audio",
        action="store_true",
        help="have the callers save each call's audio (lh-voice bots --save-audio), for the demo "
        "video; recorded in run.json",
    )
    run.add_argument(
        "--agents-seed",
        type=int,
        metavar="N",
        help="AIPerf's seed in place of the arm file's [agents] seed, as a campaign file's "
        "agents-seed line sets it for its points; run.json keeps the seed the replay ran with",
    )
    run.add_argument(
        "--kv-sample-s",
        type=_seconds,
        default=0.5,
        metavar="SECONDS",
        help="how often an agent point samples the engine's KV use and requests from its /metrics "
        "while the calls run, for the measured running share (default %(default)s); 0 samples "
        "nothing, as W5's points ran; run.json keeps the interval",
    )
    run.add_argument(
        "--profile",
        metavar="NAME",
        help="the GPU node's hardware profile (deploy/profiles/NAME.toml), whose model and scheduler "
        "settings the engine check expects; default the host's (~/.config/leaseholder/e2-host.toml), "
        "else the frozen box's",
    )
    run.add_argument(
        "--gpu-source",
        action="append",
        default=[],
        metavar="NODE=REVISION",
        help="a GPU node's deployed revision (its ~/leaseholder/src/DEPLOYED, empty when it has "
        "none), as campaign.sh reads it at the point's start; run.json keeps it beside cyan's "
        "(repeatable, one per node)",
    )
    report = actions.add_parser(
        "report",
        help="build the frontier from run directories",
        description="Write frontier.csv, frontier.md, a frontier figure per workload and the "
        "stage waterfall of the points with turn lines, waterfall.svg.",
    )
    report.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR")
    report.add_argument("--out", type=Path, required=True, help="report directory")
    report.add_argument(
        "--waterfall",
        action="append",
        default=[],
        metavar="POINT",
        help="draw only this point's configuration in waterfall.svg (repeatable, in this order; "
        "default every one with turn lines)",
    )
    report.add_argument(
        "--dots",
        type=Path,
        metavar="FILE",
        help="also draw voice-p95.svg and its phone layout voice-p95-narrow.svg: the V2V p95 and "
        "the agents' tokens/s of the configurations a TOML file names, one [[row]] each "
        '(point = "...", label = "...", note = "...", change = true for the engine change\'s rows, '
        "featured = true for the one to deploy), in its order, or [[group]] tables of rows, one "
        'per GPU (title = "...", and gpu = "..." where the GPUs ran the same points)',
    )
    export = actions.add_parser(
        "export",
        help="write the compact records of run directories, from which report gives the same tables",
        description="Write each complete point's compact record into DIR/CAMPAIGN/POINT: its "
        "run.json and arm.toml, the callers' summary.json and turns.jsonl, a summary.json with "
        "the first bytes taken exactly from the request log and the GPU's clocks, and an "
        "engine.json reduced to the engine the point ran on; no request log, AIPerf export, "
        "journal, credential or home path.",
    )
    export.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR")
    export.add_argument("--out", type=Path, required=True, help="records directory")
    engine = actions.add_parser(
        "engine",
        help="print the engine an arm file needs, as deploy/phastform/variant.sh use takes it",
    )
    engine.add_argument("arm", type=Path, help="arm file (TOML)")
    actions.add_parser("reset", help="undo what e2 run changed in cyan's units")
    e2.set_defaults(func=_e2)


def _add_load_arguments(parser: argparse.ArgumentParser) -> None:
    """The background-load flags of the latency probes."""
    parser.add_argument(
        "--background",
        type=LoadSpec.parse,
        help="agent load while measuring: prefill:<tokens>x<concurrency> (cold prefills) or "
        "decode:<tokens>x<concurrency> (long decodes)",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=1024,
        help="output tokens per decode-load request (default %(default)s)",
    )
    parser.add_argument(
        "--no-priority",
        action="store_true",
        help="omit the vLLM priority field (needed against a server without priority scheduling)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lh-bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for add in (
        _add_scale_traces,
        _add_mock_server,
        _add_kv_metrics,
        _add_tiers,
        _add_voice_turn,
        _add_e2,
        _add_drill,
    ):
        add(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
