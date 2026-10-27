use super::*;
use crate::aiperf::PlanOptions;
use crate::engine::{CpuPolicyKind, EngineConfig, GpuPolicyKind};
use crate::trace::{Trace, TraceSource};

const S: u64 = 1_000_000_000;

fn run(traces: &[&str], cfg: ReplayConfig) -> ReplayOutput {
    let engine = EngineConfig {
        gpu_blocks: 4096,
        cpu_blocks: 0,
        block_tokens: 16,
        gpu_policy: GpuPolicyKind::Lru,
        cpu_policy: CpuPolicyKind::Lru,
    };
    run_with(traces, cfg, &engine, Retention::None)
}

fn run_with(
    traces: &[&str],
    cfg: ReplayConfig,
    engine: &EngineConfig,
    retention: Retention,
) -> ReplayOutput {
    let traces = traces
        .iter()
        .map(|t| Trace::from_json(t.as_bytes()).unwrap())
        .collect();
    let mut corpus = Corpus::new(TraceSource::from_traces(traces), PlanOptions::default());
    Replay::new(
        cfg,
        &mut corpus,
        Engine::new(engine, None),
        TierCost::frozen_box(),
        Service::Idle,
        retention,
        false,
    )
    .run()
    .unwrap()
}

/// Lanes that start at their traces' first requests and play each trace once, with nothing
/// pulled forward.
fn from_start(concurrency: usize) -> ReplayConfig {
    ReplayConfig {
        concurrency,
        start_min_ratio: 0.0,
        start_max_ratio: 0.0,
        system_idle_cap_s: None,
        max_sessions: Some(concurrency as u64),
        duration_s: 3_600.0,
        grace_s: 600.0,
        ..ReplayConfig::default()
    }
}

fn find(out: &ReplayOutput, conv: u32, turn: u32) -> &RequestRecord {
    out.records
        .iter()
        .find(|r| (r.conv, r.turn) == (conv, turn))
        .expect("the request ran")
}

const PAUSES: &str = r#"{"id": "pauses", "block_size": 2, "requests": [
    {"t": 0, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [1], "api_time": 1.0},
    {"t": 600, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [1, 2], "api_time": 1.0, "think_time": 599.0},
    {"t": 640, "type": "n", "model": "m", "in": 6, "out": 1, "hash_ids": [1, 2, 3], "think_time": 30.0}
]}"#;

#[test]
fn think_times_are_capped() {
    let out = run(&[PAUSES], from_start(1));
    assert_eq!(out.records.len(), 3);
    let (t0, t1, t2) = (find(&out, 0, 0), find(&out, 0, 1), find(&out, 0, 2));
    assert_eq!(t0.issued_ns, 0);
    assert_eq!(
        t1.issued_ns - t0.done_ns,
        120 * S,
        "599 s is capped at 120 s"
    );
    assert_eq!(t2.issued_ns - t1.done_ns, 30 * S);
    assert_eq!(
        (t1.delay_ms, t2.delay_ms),
        (Some(120_000.0), Some(30_000.0))
    );
}

#[test]
fn the_idle_cap_pulls_pauses_forward() {
    let out = run(
        &[PAUSES],
        ReplayConfig {
            system_idle_cap_s: Some(10.0),
            ..from_start(1)
        },
    );
    let (t0, t1, t2) = (find(&out, 0, 0), find(&out, 0, 1), find(&out, 0, 2));
    assert_eq!(t1.issued_ns - t0.done_ns, 10 * S);
    assert_eq!(t2.issued_ns - t1.done_ns, 10 * S);
    assert_eq!(out.stats.idle_cap_jumps, 2);
    assert!((out.stats.idle_cap_skipped_s - 130.0).abs() < 1e-6);
}

#[test]
fn the_idle_cap_counters_stop_when_sending_closes() {
    const LONG: &str = r#"{"id": "long", "block_size": 2, "requests": [
        {"t": 0, "type": "n", "model": "m", "in": 2, "out": 2000, "hash_ids": [7], "api_time": 1.0}
    ]}"#;
    let capped = ReplayConfig {
        system_idle_cap_s: Some(10.0),
        ..from_start(2)
    };
    let all = run(&[PAUSES, LONG], capped.clone());
    let long = all
        .records
        .iter()
        .find(|r| r.trace == 1)
        .expect("the long request");
    // Sending closes while the long request runs and `pauses` thinks before its turn 1. When the
    // long request returns, in the grace period, the cap still pulls turn 1's dispatch forward,
    // but turn 1 is never sent: that jump is not counted.
    let close_s = long.done_ns as f64 / 2e9 - all.stats.profiling_start_s;
    let out = run(
        &[PAUSES, LONG],
        ReplayConfig {
            duration_s: close_s,
            ..capped
        },
    );
    assert_eq!(out.records.len(), 2);
    assert_eq!(
        (out.stats.idle_cap_jumps, out.stats.idle_cap_skipped_s),
        (0, 0.0)
    );
}

#[test]
fn subagents_run_together_and_the_parent_waits_for_both() {
    let trace = r#"{"id": "fanout", "block_size": 2, "requests": [
        {"t": 0, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [1], "api_time": 1.0},
        {"t": 2, "type": "subagent", "agent_id": "a", "duration_ms": 8000, "requests": [
            {"t": 2, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [9, 10], "api_time": 1.0},
            {"t": 6, "type": "n", "model": "m", "in": 6, "out": 1, "hash_ids": [9, 10, 11], "think_time": 3.0}
        ]},
        {"t": 2, "type": "subagent", "agent_id": "b", "duration_ms": 8000, "requests": [
            {"t": 2, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [20, 21], "api_time": 1.0},
            {"t": 7, "type": "n", "model": "m", "in": 6, "out": 1, "hash_ids": [20, 21, 22], "think_time": 4.0}
        ]},
        {"t": 20, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [1, 2], "think_time": 1.0}
    ]}"#;
    let out = run(&[trace], from_start(1));
    assert_eq!(out.records.len(), 6);
    let root0 = find(&out, 0, 0);
    let (a0, a1, b0, b1) = (
        find(&out, 1, 0),
        find(&out, 1, 1),
        find(&out, 2, 0),
        find(&out, 2, 1),
    );
    // Both children start when the spawning turn returns, and overlap.
    assert_eq!((a0.issued_ns, b0.issued_ns), (root0.done_ns, root0.done_ns));
    assert!(a0.issued_ns < b0.done_ns && b0.issued_ns < a0.done_ns);
    assert_eq!(
        (a1.issued_ns - a0.done_ns, b1.issued_ns - b0.done_ns),
        (3 * S, 4 * S)
    );
    assert_eq!(a0.kind, ConvKind::Subagent);
    // The parent's next turn joins both children: it goes when the last one finishes.
    assert_eq!(find(&out, 0, 1).issued_ns, a1.done_ns.max(b1.done_ns));
}

/// Session A pauses 5 s twice. During its second pause session C sends a request (leaving cached
/// blocks newer than A's), then session B one of `b_blocks` blocks, in an 11-block pool with a
/// 64-block CPU tier. Returns A's third turn under `retention`.
fn third_turn_after_pauses(retention: Retention, b_blocks: u64) -> RequestRecord {
    let a = r#"{"id": "a", "block_size": 16, "requests": [
        {"t": 0, "type": "n", "model": "m", "in": 64, "out": 1, "hash_ids": [1, 2, 3, 4], "api_time": 0.1},
        {"t": 5, "type": "n", "model": "m", "in": 80, "out": 1, "hash_ids": [1, 2, 3, 4, 5], "think_time": 5.0},
        {"t": 10, "type": "n", "model": "m", "in": 96, "out": 1, "hash_ids": [1, 2, 3, 4, 5, 6], "think_time": 5.0}
    ]}"#;
    let ids: Vec<String> = (101..101 + b_blocks).map(|i| i.to_string()).collect();
    let b = format!(
        r#"{{"id": "b", "block_size": 16, "requests": [
            {{"t": 0, "type": "n", "model": "m", "in": 16, "out": 1, "hash_ids": [100], "api_time": 0.1}},
            {{"t": 8, "type": "n", "model": "m", "in": {}, "out": 1, "hash_ids": [{}], "think_time": 8.0}}
        ]}}"#,
        16 * b_blocks,
        ids.join(",")
    );
    let c = r#"{"id": "c", "block_size": 16, "requests": [
        {"t": 0, "type": "n", "model": "m", "in": 16, "out": 1, "hash_ids": [200], "api_time": 0.1},
        {"t": 6.5, "type": "n", "model": "m", "in": 48, "out": 1, "hash_ids": [200, 201, 202], "think_time": 6.5}
    ]}"#;
    let engine = EngineConfig {
        gpu_blocks: 12,
        cpu_blocks: 64,
        block_tokens: 16,
        gpu_policy: GpuPolicyKind::Lru,
        cpu_policy: CpuPolicyKind::Lru,
    };
    let out = run_with(&[a, &b, c], from_start(3), &engine, retention);
    assert_eq!(out.records.len(), 7);
    out.records
        .iter()
        .find(|r| (r.trace, r.turn) == (0, 2))
        .expect("A's third turn ran")
        .clone()
}

fn sources(r: &RequestRecord) -> (u64, u64) {
    (r.sources.gpu, r.sources.cpu)
}

#[test]
fn a_ttl_hold_keeps_a_paused_session_over_newer_cached_blocks() {
    // LRU evicts A's newest block for B's 4 blocks, before C's newer ones; the hold keeps it.
    assert_eq!(
        sources(&third_turn_after_pauses(Retention::None, 4)),
        (64, 16)
    );
    assert_eq!(
        sources(&third_turn_after_pauses(Retention::Ttl(30.0), 4)),
        (80, 0)
    );
    // A hold shorter than the pause has ended by then.
    assert_eq!(
        sources(&third_turn_after_pauses(Retention::Ttl(1.0), 4)),
        (64, 16)
    );
}

#[test]
fn a_prefetch_brings_an_evicted_session_back_before_it_returns() {
    // B's 11 blocks evict everything; A's blocks come back from the CPU tier.
    assert_eq!(
        sources(&third_turn_after_pauses(Retention::None, 11)),
        (0, 80)
    );
    for retention in [
        Retention::OraclePrefetch,
        // A's first pause predicts its second exactly.
        Retention::HistoryPrefetch,
        Retention::Tokencake,
    ] {
        let r = third_turn_after_pauses(retention, 11);
        assert_eq!(sources(&r), (80, 0), "{retention:?}");
    }
}

#[test]
fn the_oracle_prefetch_moves_with_a_turn_the_idle_cap_pulls_forward() {
    // A pauses 60 s. B's second turn evicts A's blocks from an 11-block pool; then nothing is in
    // flight, and with the cap A's next turn goes 10 s after B's returns.
    let a = r#"{"id": "a", "block_size": 16, "requests": [
        {"t": 0, "type": "n", "model": "m", "in": 64, "out": 1, "hash_ids": [1, 2, 3, 4], "api_time": 0.1},
        {"t": 60, "type": "n", "model": "m", "in": 80, "out": 1, "hash_ids": [1, 2, 3, 4, 5], "think_time": 60.0}
    ]}"#;
    let ids: Vec<String> = (101..112).map(|i| i.to_string()).collect();
    let b = format!(
        r#"{{"id": "b", "block_size": 16, "requests": [
            {{"t": 0, "type": "n", "model": "m", "in": 16, "out": 1, "hash_ids": [100], "api_time": 0.1}},
            {{"t": 1, "type": "n", "model": "m", "in": 176, "out": 1, "hash_ids": [{}], "think_time": 1.0}}
        ]}}"#,
        ids.join(",")
    );
    let engine = EngineConfig {
        gpu_blocks: 12,
        cpu_blocks: 64,
        block_tokens: 16,
        gpu_policy: GpuPolicyKind::Lru,
        cpu_policy: CpuPolicyKind::Lru,
    };
    for cap in [None, Some(10.0)] {
        let cfg = ReplayConfig {
            system_idle_cap_s: cap,
            ..from_start(2)
        };
        for (retention, expected) in [
            (Retention::None, (0, 64)),
            (Retention::OraclePrefetch, (64, 0)),
        ] {
            let out = run_with(&[a, &b], cfg.clone(), &engine, retention);
            let turn = |trace, turn| {
                out.records
                    .iter()
                    .find(|r| (r.trace, r.turn) == (trace, turn))
                    .expect("the turn ran")
            };
            let (a0, a1, b1) = (turn(0, 0), turn(0, 1), turn(1, 1));
            let pause = match cap {
                None => 60 * S,
                Some(_) => b1.done_ns + 10 * S - a0.done_ns,
            };
            assert_eq!(a1.issued_ns - a0.done_ns, pause, "{cap:?}");
            assert_eq!(sources(a1), expected, "{retention:?}, cap {cap:?}");
        }
    }
}

#[test]
fn a_drained_lane_recycles_under_a_fresh_namespace() {
    let ids: Vec<String> = (1..=16).map(|i| i.to_string()).collect();
    let trace = format!(
        r#"{{"id": "loop", "block_size": 2, "requests": [
            {{"t": 0, "type": "n", "model": "m", "in": 16, "out": 1, "hash_ids": [{}], "api_time": 1.0}},
            {{"t": 5, "type": "n", "model": "m", "in": 32, "out": 1, "hash_ids": [{}], "think_time": 1.0}}
        ]}}"#,
        ids[..8].join(","),
        ids.join(",")
    );
    let out = run(
        &[&trace],
        ReplayConfig {
            max_sessions: Some(3),
            ..from_start(1)
        },
    );
    assert_eq!((out.stats.sessions_started, out.records.len()), (3, 6));
    let mut last_done = 0;
    for play in out.records.chunks(2) {
        let (first, second) = (&play[0], &play[1]);
        assert_eq!((first.turn, second.turn), (0, 1));
        // The prompt the previous play cached is computed again: its blocks are new.
        assert_eq!(
            (first.sources.gpu, first.sources.computed, first.recomputed),
            (0, 16, 0)
        );
        // The second turn hits the first turn's block within the play.
        assert_eq!(second.sources.gpu, 16);
        assert!(
            first.issued_ns >= last_done,
            "a lane recycles only after its tree drains"
        );
        last_done = second.done_ns;
    }
}

#[test]
fn the_realtime_tier_is_short_while_too_little_of_the_pool_is_free() {
    // 11 usable blocks: an 8-block prompt, a 1 s pause, then a 9-block prompt over it.
    let trace = r#"{"id": "rt", "block_size": 16, "requests": [
        {"t": 0, "type": "n", "model": "m", "in": 128, "out": 1, "hash_ids": [1, 2, 3, 4, 5, 6, 7, 8], "api_time": 0.1},
        {"t": 2, "type": "n", "model": "m", "in": 144, "out": 1, "hash_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "think_time": 1.0}
    ]}"#;
    let engine = EngineConfig {
        gpu_blocks: 12,
        cpu_blocks: 0,
        block_tokens: 16,
        gpu_policy: GpuPolicyKind::Lru,
        cpu_policy: CpuPolicyKind::Lru,
    };
    let run_rt = |realtime_tokens| {
        let cfg = ReplayConfig {
            realtime_tokens,
            ..from_start(1)
        };
        run_with(&[trace], cfg, &engine, Retention::None)
    };
    let share = |ns: u64, end: u64| ns as f64 / end as f64;
    // 4 blocks: the working set leaves fewer free throughout, the pinned prompt while each runs.
    let out = run_rt(64);
    let (first, second) = (&out.records[0], &out.records[1]);
    let end = second.done_ns;
    let running = first.done_ns - first.admitted_ns + second.done_ns - second.admitted_ns;
    assert_eq!(out.stats.realtime_short_share, Some(1.0));
    let blocked = out.stats.realtime_blocked_share.unwrap();
    assert!((blocked - share(running, end)).abs() < 1e-12, "{blocked}");
    // 3 blocks fit beside the 8-block prompt, pinned or not, but not beside the 9-block one.
    let out = run_rt(48);
    let second = &out.records[1];
    assert_eq!(second.done_ns, end);
    let short = out.stats.realtime_short_share.unwrap();
    assert!((short - share(end - second.issued_ns, end)).abs() < 1e-12);
    let blocked = out.stats.realtime_blocked_share.unwrap();
    assert!((blocked - share(second.done_ns - second.admitted_ns, end)).abs() < 1e-12);
    let none = run_rt(0).stats;
    assert_eq!(
        (none.realtime_short_share, none.realtime_blocked_share),
        (None, None)
    );
}
