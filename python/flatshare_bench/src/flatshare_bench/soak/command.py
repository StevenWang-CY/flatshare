"""``flatshare-bench soak``'s command line: ``run``, ``report`` and ``export`` (:mod:`flatshare_bench.soak`). Each
action's function imports what it runs, as the other commands' do (:mod:`flatshare_bench.cli`), so that this module
loads nothing but argparse."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path


def _seconds(text: str) -> float:
    """An interval in seconds (or hours), 0 or more, for argparse."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds") from None
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r}: an interval is 0 or more seconds")
    return value


def _soak_run(args: argparse.Namespace) -> int:
    from flatshare_bench.e2.cyan import Host
    from flatshare_bench.e2.run import gpu_sources
    from flatshare_bench.soak.guard import parse_guards
    from flatshare_bench.soak.run import SoakOptions, run_soak
    from flatshare_bench.soak.summary import render_hour

    # A closed ssh session does not stop a soak; SIGINT and SIGTERM end it, its summary written.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    options = SoakOptions(
        hours=args.hours,
        slo_reference_ms=args.slo_reference,
        slo_margin_ms=args.slo_margin,
        calls_per_hour=args.calls_per_hour,
        turns=args.turns,
        warmup_turns=args.warmup_turns,
        barge_in=args.barge_in,
        caller_seed=args.caller_seed,
        dispatch_timeout_s=args.dispatch_timeout,
        max_open=args.max_open,
        agents_sessions=args.agents_sessions,
        agents_data=args.agents_data,
        agents_seed=args.agents_seed,
        think_cap_s=args.think_cap,
        agents_warmup_s=args.agents_warmup,
        agents_run_hours=args.agents_run_hours,
        scrape_s=args.scrape_s,
        guard_s=args.guard_s,
        guards=parse_guards(args.guard),
        guard_clear_s=args.guard_clear_s,
        drain_s=args.drain_s,
        model=args.model,
        gpu_source=gpu_sources(args.gpu_source) if args.gpu_source else None,
    )
    try:
        summary = run_soak(options, args.out, Host.load())
    except KeyboardInterrupt:
        print(f"flatshare-bench soak: {args.out} aborted", file=sys.stderr)
        return 130
    if summary["overall"] is not None:
        print(f"{args.out}: {render_hour(summary['overall'])}")
    return 0


def _soak_report(args: argparse.Namespace) -> int:
    from flatshare_bench.soak.summary import load, render_markdown, summarize, write_summary

    data = load(args.soak)
    reference = args.slo_reference
    if reference is None:
        reference = (data.record.get("options") or {}).get("slo_reference_ms")
    summary = summarize(data, reference, args.slo_margin)
    if args.write:
        write_summary(args.soak, summary)
    print(render_markdown(summary), end="")
    return 0


def _soak_export(args: argparse.Namespace) -> int:
    from flatshare_bench.soak.summary import export

    try:
        print(export(args.soak, args.out, args.slo_reference, args.slo_margin))
    except ValueError as exc:
        print(f"flatshare-bench soak: {exc}", file=sys.stderr)
        return 1
    return 0


def _positive_float(text: str) -> float:
    """A number above 0, for argparse."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r}: must be above 0")
    return value


def _count(text: str) -> int:
    """A count, 0 or more, for argparse."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r}: must be 0 or more")
    return value


def add_soak(commands: argparse._SubParsersAction) -> None:
    soak = commands.add_parser(
        "soak",
        help="soak the production path for hours: calls, agents, scrapes, a guard; every attempt kept",
        description="Load the box as it serves, for hours, and record every attempt and every scrape: calls arriving "
        "on a schedule against the voice worker in serve mode, AIPerf's agentic replay through the gateway, the "
        "gateway's and engines' metrics every minute, cyan's units' restarts and failures, and a guard that stops "
        "the load while another's process holds a GPU (ops/RUNBOOK.md, Soaks).",
    )
    actions = soak.add_subparsers(dest="action", required=True)
    run = actions.add_parser(
        "run",
        help="run a soak on cyan",
        description="Check that the box serves as deployed, then run the soak's calls and agents for --hours, "
        "scraping the service and guarding the GPUs; writes the soak directory and its summary per hour and "
        "overall. SIGINT or SIGTERM ends it, the summary written; one runs at a time, under E2's lock.",
    )
    run.add_argument(
        "--hours", type=_positive_float, required=True, help="the soak's window, in hours"
    )
    run.add_argument("--out", type=Path, required=True, help="the soak directory")
    slo = run.add_argument_group("the SLO")
    slo.add_argument(
        "--slo-reference",
        type=_positive_float,
        metavar="MS",
        help="voice alone's V2V p95 in ms with the same calls on the same box, the SLO's reference; the summary "
        "counts the replies over it plus --slo-margin (none: soak report --slo-reference adds it later)",
    )
    slo.add_argument(
        "--slo-margin",
        type=_positive_float,
        default=250.0,
        metavar="MS",
        help="what agent load may add to the reference (default %(default)s, docs/SLO.md)",
    )
    calls = run.add_argument_group("the calls (flatshare-voice arrivals)")
    calls.add_argument(
        "--calls-per-hour",
        type=_positive_float,
        default=20.0,
        help="the calls' mean arrival rate, a Poisson process (default %(default)s)",
    )
    calls.add_argument(
        "--turns", type=int, default=10, help="measured turns per call (default %(default)s)"
    )
    calls.add_argument(
        "--warmup-turns",
        type=_count,
        default=1,
        help="unmeasured turns that open each call (default %(default)s: a serving worker's AEC warm-up "
        "ignores an interruption of its first reply)",
    )
    calls.add_argument(
        "--barge-in",
        type=float,
        default=0.25,
        help="share of turns with an interruption (default %(default)s)",
    )
    calls.add_argument(
        "--caller-seed", type=int, default=0, help="the calls' seed (default %(default)s)"
    )
    calls.add_argument(
        "--dispatch-timeout",
        type=_positive_float,
        default=15.0,
        metavar="SECONDS",
        help="how long a caller waits for the agent before the call counts as refused (default %(default)s)",
    )
    calls.add_argument(
        "--max-open",
        type=int,
        default=8,
        help="attempts open at once at most; a call due beyond them is skipped (default %(default)s)",
    )
    calls.add_argument(
        "--drain-s",
        type=_seconds,
        default=600.0,
        metavar="SECONDS",
        help="how long the calls in progress at the window's end may play on (default %(default)s)",
    )
    agents = run.add_argument_group("the agents (AIPerf's agentic replay, as E2 runs it)")
    agents.add_argument(
        "--agents-sessions",
        type=_count,
        default=16,
        help="session trees AIPerf keeps live; 0 runs voice alone (default %(default)s)",
    )
    agents.add_argument(
        "--agents-data",
        default="weka-x32",
        help="the corpus, a scale-traces directory under ~/flatshare/data (default %(default)s)",
    )
    agents.add_argument(
        "--agents-seed", type=int, default=11, help="AIPerf's seed (default %(default)s)"
    )
    agents.add_argument(
        "--think-cap",
        type=_positive_float,
        default=120.0,
        metavar="SECONDS",
        help="cap on each recorded think time (default %(default)s)",
    )
    agents.add_argument(
        "--agents-warmup",
        type=_seconds,
        default=60.0,
        metavar="SECONDS",
        help="seconds of AIPerf's profiling before the window starts (default %(default)s)",
    )
    agents.add_argument(
        "--agents-run-hours",
        type=_seconds,
        default=0.0,
        metavar="HOURS",
        help="start AIPerf again after this many hours of a run; 0: one run for the window (default %(default)s)",
    )
    agents.add_argument(
        "--model",
        help="the model AIPerf asks for (default the serving profile's, else the GPU node's hardware profile's)",
    )
    service = run.add_argument_group("the scrapes and the guard")
    service.add_argument(
        "--scrape-s",
        type=_positive_float,
        default=60.0,
        metavar="SECONDS",
        help="how often the service is scraped (default %(default)s)",
    )
    service.add_argument(
        "--guard-s",
        type=_positive_float,
        default=60.0,
        metavar="SECONDS",
        help="how often each GPU node is checked (default %(default)s)",
    )
    service.add_argument(
        "--guard",
        action="append",
        default=[],
        metavar="NODE=METHOD",
        help="how a node's GPU is checked: ssh (the default: the guard key at its tunnel's target, "
        "deploy/authorize-guard.sh), ssh:TARGET, local (this host's GPU; a single host's default) or off "
        "(repeatable, a node each)",
    )
    service.add_argument(
        "--guard-clear-s",
        type=_seconds,
        default=300.0,
        metavar="SECONDS",
        help="how long every GPU must stay clear before a paused load resumes (default %(default)s)",
    )
    service.add_argument(
        "--gpu-source",
        action="append",
        default=[],
        metavar="NODE=REVISION",
        help="a GPU node's deployed revision, which soak.json keeps (repeatable)",
    )
    run.set_defaults(func=_soak_run)
    report = actions.add_parser(
        "report",
        help="the summary of a soak directory or its compact record",
        description="Print a soak's summary per hour and overall as a table, from its directory or its compact "
        "record; --write also writes summary.json and summary.md beside it.",
    )
    report.add_argument("soak", type=Path, help="a soak directory, or its compact record")
    report.add_argument(
        "--slo-reference",
        type=_positive_float,
        metavar="MS",
        help="voice alone's V2V p95 in ms (default the soak's own --slo-reference)",
    )
    report.add_argument("--slo-margin", type=_positive_float, default=250.0, metavar="MS")
    report.add_argument("--write", action="store_true", help="write summary.json and summary.md")
    report.set_defaults(func=_soak_report)
    export = actions.add_parser(
        "export",
        help="write a soak's compact record, to commit",
        description="Write the compact record of a soak directory into OUT: soak.json, attempts.jsonl, turns.csv, "
        "series.csv (a column a kept series), events.jsonl, units.jsonl, guard.csv, summary.json and summary.md; "
        "no raw scrape, AIPerf export, request log or credential, and paths under a home directory from ~.",
    )
    export.add_argument("soak", type=Path, help="a soak directory")
    export.add_argument("--out", type=Path, required=True, help="the record's directory")
    export.add_argument(
        "--slo-reference",
        type=_positive_float,
        metavar="MS",
        help="voice alone's V2V p95 in ms (default the soak's own --slo-reference)",
    )
    export.add_argument("--slo-margin", type=_positive_float, default=250.0, metavar="MS")
    export.set_defaults(func=_soak_export)
