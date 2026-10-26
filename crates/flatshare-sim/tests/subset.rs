//! End to end on the 16-trace subset in `testdata/`: the AIPerf port against AIPerf's own replay
//! structure, a run of every stock arm, of the E1 arms and of v0's idle service, and `replay-log` on
//! a simulated run and on a record of AIPerf's own.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use flatshare_core::{
    FROZEN_BOX_POOL_TOKENS, KvGeometry, PromptSources, StepCost, TierCapacity, TierCost,
};
use flatshare_sim::aiperf::{PlanOptions, reference};
use flatshare_sim::corpus::Corpus;
use flatshare_sim::engine::Engine;
use flatshare_sim::forecast;
use flatshare_sim::hardware::{self, A100_40GB, Gpu, H100_SXM, RTX_5060_TI};
use flatshare_sim::replay::{Pauses, Replay, ReplayConfig, Service, TimelineEvent};
use flatshare_sim::retention::Retention;
use flatshare_sim::run::{self, Arm, ArmResult, RunConfig};
use flatshare_sim::{regime, replay_log};
use serde_json::{Value, json};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

fn corpus() -> Corpus {
    Corpus::open(&testdata("weka-x32-sub16"), PlanOptions::default()).unwrap()
}

/// Collects the differences between two JSON values; numbers agree to 1e-9 relative.
fn diff(path: &str, a: &Value, b: &Value, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys().filter(|k| !x.contains_key(*k))) {
                diff(
                    &format!("{path}.{k}"),
                    x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null),
                    out,
                );
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                out.push(format!("{path}: {} vs {} items", x.len(), y.len()));
            }
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                diff(&format!("{path}[{i}]"), u, v, out);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() > 1e-9 * x.abs().max(1.0) {
                out.push(format!("{path}: {x} vs {y}"));
            }
        }
        _ if a != b => out.push(format!("{path}: {a} vs {b}")),
        _ => {}
    }
}

#[test]
fn the_replay_structure_matches_aiperf() {
    // testdata/aiperf-0.13.0-sub16.json is AIPerf's own plan (scripts/aiperf_reference.py): 16 lanes,
    // seed 11, think times capped at 120 s, start ratios 0 to 1.
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(testdata("aiperf-0.13.0-sub16.json")).unwrap(),
    )
    .unwrap();
    let mut c = corpus();
    let plans: Vec<_> = (0..c.len()).map(|i| c.plan(i).unwrap()).collect();
    let refs: Vec<&_> = plans.iter().map(|p| p.as_ref()).collect();
    let ours = reference::dump(&refs, 16, 11, 0.0, 1.0);
    let by_id = |v: &Value| -> HashMap<String, Value> {
        v["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| (c["id"].as_str().unwrap().to_owned(), c.clone()))
            .collect()
    };
    let (theirs, mine) = (by_id(&golden), by_id(&ours));
    assert_eq!(theirs.len(), 98);
    let mut problems = Vec::new();
    for (id, conv) in &theirs {
        match mine.get(id) {
            Some(m) => diff(id, conv, m, &mut problems),
            None => problems.push(format!("{id}: missing")),
        }
    }
    problems.extend(
        mine.keys()
            .filter(|id| !theirs.contains_key(*id))
            .map(|id| format!("{id}: not in AIPerf's plan")),
    );
    diff("lanes", &golden["lanes"], &ours["lanes"], &mut problems);
    diff(
        "recycle",
        &golden["recycle"],
        &ours["recycle"],
        &mut problems,
    );
    assert!(
        problems.is_empty(),
        "{} differences, first: {:?}",
        problems.len(),
        &problems[..problems.len().min(10)]
    );
}

fn run_all(seed: u64) -> Vec<ArmResult> {
    let cfg = RunConfig::frozen_box(ReplayConfig {
        seed,
        ..ReplayConfig::default()
    });
    let capacity = TierCapacity::frozen_box(&cfg.geometry);
    run::run(&mut corpus(), &cfg, &Arm::STOCK, &[capacity], |_| {}).unwrap()
}

#[test]
fn every_arm_runs_consistently_on_the_subset() {
    let results = run_all(11);
    let names: Vec<_> = results.iter().map(|r| r.summary.arm.as_str()).collect();
    assert_eq!(names, ["gpu-lru", "lru-cpu", "clairvoyant"]);
    let bytes_per_token = flatshare_core::KvGeometry::QWEN3_4B_BF16.bytes_per_token;
    for r in &results {
        let s = &r.summary;
        assert!(s.requests > 100, "{}: {} requests", s.arm, s.requests);
        assert_eq!(s.requests as usize, r.records.len());
        assert_eq!(s.mix.main + s.mix.subagent + s.mix.flattened, s.requests);
        assert_eq!(
            s.gpu_tokens + s.cpu_tokens + s.computed_tokens,
            s.prompt_tokens
        );
        assert_eq!(
            s.prompt_tokens,
            r.records.iter().map(|x| x.prompt_tokens).sum::<u64>()
        );
        let sh = &s.shares;
        assert!(
            (sh.gpu + sh.cpu + sh.computed - 1.0).abs() < 1e-9,
            "{}: shares {sh:?}",
            s.arm
        );
        assert!((sh.first_turn + sh.new_context + sh.recomputed - sh.computed).abs() < 1e-9);
        assert!(
            [sh.first_turn, sh.new_context, sh.recomputed]
                .iter()
                .all(|&x| x >= 0.0)
        );
        assert_eq!(s.pcie_restored_bytes, s.cpu_tokens * bytes_per_token);
        assert!(s.ttft_ms.p50 <= s.ttft_ms.p90 && s.ttft_ms.p90 <= s.ttft_ms.p99);
        assert!(s.regime.kv_pressure_peak >= s.regime.kv_pressure_mean && s.regime.rho > 0.0);
        assert_eq!(s.replay.lanes, 16);
    }
    let [gpu_lru, lru_cpu, clairvoyant] = &results[..] else {
        unreachable!()
    };
    assert_eq!(
        (gpu_lru.summary.cpu_blocks, gpu_lru.summary.cpu_tokens),
        (0, 0)
    );
    assert!(lru_cpu.summary.cpu_tokens > 0);
    // The clairvoyant reference re-prices the requests of the lru-cpu run, on its timeline.
    let c = &clairvoyant.summary;
    assert_eq!(c.reference.as_deref(), Some("lru-cpu"));
    assert_eq!(
        (c.requests, c.prompt_tokens, c.stalls),
        (
            lru_cpu.summary.requests,
            lru_cpu.summary.prompt_tokens,
            lru_cpu.summary.stalls
        )
    );
    for (x, y) in clairvoyant.records.iter().zip(&lru_cpu.records) {
        assert_eq!(
            (x.conv, x.turn, x.issued_ns, x.admitted_ns),
            (y.conv, y.turn, y.issued_ns, y.admitted_ns)
        );
    }
}

#[test]
fn a_fixed_seed_gives_the_same_run() {
    let (a, b) = (run_all(5), run_all(5));
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.summary, y.summary);
        assert_eq!(x.records, y.records);
    }
    // Another seed starts the lanes at other points of their traces.
    let other = run_all(6);
    assert_ne!(other[1].summary.prompt_tokens, a[1].summary.prompt_tokens);
}

#[test]
fn the_e1_arms_run_consistently_and_repeat_exactly() {
    let cfg = RunConfig::frozen_box(ReplayConfig {
        pauses: Pauses::TraceLab { cap_s: Some(120.0) },
        ..ReplayConfig::default()
    });
    let capacity = TierCapacity::frozen_box(&cfg.geometry);
    let run_e1 = || run::run(&mut corpus(), &cfg, &Arm::E1, &[capacity], |_| {}).unwrap();
    let (a, b) = (run_e1(), run_e1());
    let names: Vec<_> = a.iter().map(|r| r.summary.arm.as_str()).collect();
    assert_eq!(names, Arm::E1.map(Arm::name));
    let block_bytes = flatshare_core::KvGeometry::QWEN3_4B_BF16.block_bytes();
    for (x, y) in a.iter().zip(&b) {
        let s = &x.summary;
        assert_eq!(s, &y.summary);
        assert!(s.requests > 100, "{}: {} requests", s.arm, s.requests);
        assert_eq!(
            s.gpu_tokens + s.cpu_tokens + s.computed_tokens,
            s.prompt_tokens,
            "{}",
            s.arm
        );
        assert_eq!(
            s.pcie_prefetched_bytes,
            s.engine.prefetched_blocks * block_bytes
        );
        // Every pause is a TraceLab draw: the median is a few hundred ms.
        assert!(
            (100.0..1_000.0).contains(&s.regime.median_think_ms),
            "{}: {}",
            s.arm,
            s.regime.median_think_ms
        );
    }
    let by_name = |n: &str| &a.iter().find(|r| r.summary.arm == n).unwrap().summary;
    assert_eq!(by_name("lru-cpu").engine.prefetched_blocks, 0);
    assert!(by_name("oracle-prefetch").engine.prefetched_blocks > 0);
    assert!(by_name("tokencake").engine.prefetched_blocks > 0);
}

#[test]
fn the_box_places_as_the_regime_sweep_runs_it_and_an_h100_elsewhere() {
    let qwen = KvGeometry::QWEN3_4B_BF16;
    let traces = testdata("weka-x32-sub16");
    let replay = ReplayConfig {
        realtime_tokens: 2 * hardware::TOKENS_PER_CALL,
        ..ReplayConfig::default()
    };
    let cell = |gpu: Gpu| regime::Cell {
        pauses: Pauses::Trace,
        sessions: 8,
        cpu_ratio: 4.0,
        capacity: gpu.capacity(&qwen, 4.0),
    };
    let cells = [RTX_5060_TI, H100_SXM].map(|gpu| hardware::Cell {
        gpu,
        token_scale: 1,
        seed: 11,
        regime: cell(gpu),
    });
    let placed = hardware::sweep(&traces, PlanOptions::default(), &replay, &cells, 2).unwrap();
    let base = RunConfig::frozen_box(replay);
    let frozen_box = regime::Cell {
        capacity: TierCapacity::with_cpu_ratio(&qwen, FROZEN_BOX_POOL_TOKENS, 4.0),
        ..cell(RTX_5060_TI)
    };
    let mapped = regime::sweep(
        &traces,
        PlanOptions::default(),
        &base,
        &[frozen_box],
        &[11],
        1,
    )
    .unwrap();
    assert_eq!(placed[0].result.arms, mapped[0].arms);
    // The H100's pool is 13.8 times the box's and its restores are 8 times faster per token.
    let (b, h) = (
        placed[0].result.arm(Arm::LruCpu),
        placed[1].result.arm(Arm::LruCpu),
    );
    assert!(h.regime.kv_pressure_mean < b.regime.kv_pressure_mean / 5.0);
    assert!(
        h.regime.pi > 2.0 * b.regime.pi,
        "{} vs {}",
        h.regime.pi,
        b.regime.pi
    );
    assert!(h.resume_ttft_ms.p50 < b.resume_ttft_ms.p50);
    let share = |s: &flatshare_sim::metrics::Summary| s.replay.realtime_blocked_share.unwrap();
    assert!(share(h) <= share(b) && share(b) < 1.0);
    let table = hardware::csv(&placed);
    assert_eq!(table.lines().count(), 1 + 2 * Arm::E1.len());
}

#[test]
fn the_forecast_places_its_cells_as_run_places_them() {
    let traces = testdata("weka-x32-sub16");
    let base = ReplayConfig {
        duration_s: 120.0,
        ..ReplayConfig::default()
    };
    let cells = forecast::cells(&A100_40GB, None);
    // The core's three cells and the calibration point's two, on the subset as both corpora.
    let cells = &cells[..5];
    let placed = forecast::place(
        &traces,
        &traces,
        PlanOptions::default(),
        &base,
        cells,
        &[11],
        2,
    )
    .unwrap();
    // The box's own core cell is `lru-cpu` at 24 sessions on the frozen box with two calls of
    // realtime tier, as `run` runs it.
    let cfg = RunConfig::frozen_box(ReplayConfig {
        concurrency: 24,
        ..forecast::replay(&base)
    });
    let capacity = TierCapacity::frozen_box(&KvGeometry::QWEN3_4B_BF16);
    let (alone, _) = run::run_arm(&mut corpus(), &cfg, Arm::LruCpu, capacity, false).unwrap();
    let s = &alone.summary;
    assert_eq!(placed[0].pressure_mean, s.regime.kv_pressure_mean);
    assert_eq!(placed[0].resume_p90_ms, s.resume_ttft_ms.p90);
    assert_eq!(
        placed[0].running_share,
        s.replay.realtime_blocked_share.unwrap()
    );
    // The A100 runs the same agents faster: less of its time busy, never a longer resume tail.
    let (b, a) = (&placed[0], &placed[2]);
    assert!(a.busy < b.busy, "{} vs {}", a.busy, b.busy);
    assert!(a.resume_p90_ms <= b.resume_p90_ms);
    let report = forecast::report(&A100_40GB, None, cells, &placed, &[11], None);
    for heading in [
        "# Forecast: the A100 40 GB against the frozen box",
        "## Step and tier costs, Qwen3-4B-Instruct-2507-FP8",
        "## Step and tier costs, Qwen3-14B-FP8",
        "## E0's cells",
        "## The engine probes",
        "## E2's scenarios on the regime map",
        "## What the forecast says the run will show",
    ] {
        assert!(report.contains(heading), "{heading}\n{report}");
    }
    assert!(report.contains("| A: the core, the box's pool | RTX 5060 Ti 16 GB (measured) | 24 |"));
}

/// `flatshare-sim forecast` for the A100 80 GB with `args`, on the subset as both corpora at seed 11 in
/// 120 s replays: the report it prints.
fn forecast_on_subset(args: &[&str]) -> String {
    let traces = testdata("weka-x32-sub16");
    let out = Command::new(env!("CARGO_BIN_EXE_flatshare-sim"))
        .arg("forecast")
        .args([&traces, &traces])
        .args(["--gpu", "a100-80gb", "--seeds", "11", "--duration", "120"])
        .args(["--jobs", "2"])
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// `testdata/forecast-a100-80gb-sub16.md` is that report as `flatshare-sim forecast` wrote it before it
/// took a card's fitted costs (`flatshare-sim forecast testdata/weka-x32-sub16 testdata/weka-x32-sub16
/// --gpu a100-80gb --seeds 11 --duration 120 --markdown testdata/forecast-a100-80gb-sub16.md`
/// rewrites it).
#[test]
fn the_forecast_is_unchanged_without_fitted_costs_and_the_boxs_own_give_its_rows() {
    let golden = std::fs::read_to_string(testdata("forecast-a100-80gb-sub16.md")).unwrap();
    assert_eq!(forecast_on_subset(&[]), golden);
    // The box's own costs as the card's fitted ones: its E0 fits as `flatshare-bench tiers` writes them,
    // and its step-time model as fit_step_cost.py does.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let (summary, steps) = (dir.join("box-e0-summary.json"), dir.join("box-steps.json"));
    let tier = TierCost::frozen_box();
    let fits = json!({"fits": {
        "gpu": tier.gpu_hit,
        "cpu_over_gpu": tier.cpu_restore,
        "recompute": tier.recompute,
    }});
    std::fs::write(&summary, fits.to_string()).unwrap();
    std::fs::write(&steps, json!(StepCost::frozen_box()).to_string()).unwrap();
    let report = forecast_on_subset(&[
        "--costs",
        summary.to_str().unwrap(),
        "--step-costs",
        steps.to_str().unwrap(),
    ]);
    // Every table is as it was, with the fitted section between the scenario table and the reading.
    let start = report.find("\n## With the card's fitted costs\n").unwrap();
    let end = report
        .find("\n## What the forecast says the run will show\n")
        .unwrap();
    assert_eq!(format!("{}{}", &report[..start], &report[end..]), golden);
    let section = &report[start..end];
    assert!(
        section.contains("`box-steps.json` and E0's tier fits in `box-e0-summary.json`."),
        "{section}"
    );
    // The box's costs give the box's rows: the calibration point's, and the core's at 24 sessions.
    let measured: Vec<_> = golden
        .lines()
        .filter(|l| l.contains(" | RTX 5060 Ti 16 GB (measured) | "))
        .collect();
    assert_eq!(measured.len(), 2);
    for row in measured {
        let fitted = row.replace("RTX 5060 Ti 16 GB (measured)", "A100 SXM4 80 GB (fitted)");
        assert!(section.contains(&fitted), "{fitted}\n{section}");
    }
    assert_eq!(section.matches(" (fitted) | ").count(), 6);
    assert!(
        section.contains("Closest to the box's running share in the core (0%): 24 sessions (0%)."),
        "{section}"
    );
}

/// `text`'s section under `heading`, to the next one.
fn section<'a>(text: &'a str, heading: &str) -> &'a str {
    let start = text.find(&format!("## {heading}\n")).expect(heading);
    let end = text[start..]
        .find("\n## ")
        .map_or(text.len(), |n| start + n);
    &text[start..end]
}

/// On the box's own host (`testdata/box-host.json`, bench_host.py's host.json on Phastform) the A100
/// 80 GB keeps the box's host time and takes the box's measured link in place of its datasheet one:
/// the report is the one without a host but for the introduction, which names the host, the costs
/// tables' rows on it, E0's restores and the card's scenario cells.
#[test]
fn on_the_boxs_own_host_the_card_keeps_the_boxs_host_time_and_takes_its_link() {
    let golden = std::fs::read_to_string(testdata("forecast-a100-80gb-sub16.md")).unwrap();
    let report = forecast_on_subset(&["--host", testdata("box-host.json").to_str().unwrap()]);
    for text in [
        "(`make forecast-a100 GPU=a100-80gb HOST_JSON=box-host.json`) from the box's measured costs, the \
         card's published peaks, the path vLLM 0.32.0 takes for the box's checkpoint on it (Marlin \
         W8A16; `crates/flatshare-sim/src/hardware.rs`) and the host it is placed on, as \
         `benchmarks/a100/bench_host.py` measured it: its CPU (Intel(R) Xeon(R) CPU E5-2609 v4 @ \
         1.70GHz) runs the Python workload in 381.1 ms against the box's 381.1, and its link copies \
         pinned host memory to the card at 6.76 GB/s in one copy against the box's 6.76. Every number",
        "On this host the API server's time is the box's over 1.00, and a restore's transfer runs at \
         92% of the host's one-copy rate, as the box's E0 restore does of the box's;",
    ] {
        assert!(report.contains(text), "{text}\n{report}");
    }
    // The probes are the steps', which a host does not change.
    let probes = "The engine probes";
    assert_eq!(section(&report, probes), section(&golden, probes));
    // Each costs table keeps its rows and adds the card's on this host: the box's 23.64 µs a token of
    // restore for the 4B's 144 KiB (26.27 for the 14B's 160 KiB), and the rest as they were.
    for (model, restore) in [
        ("Qwen3-4B-Instruct-2507-FP8", "23.64"),
        ("Qwen3-14B-FP8", "26.27"),
    ] {
        let heading = format!("Step and tier costs, {model}");
        let (ours, theirs) = (section(&report, &heading), section(&golden, &heading));
        assert!(ours.starts_with(theirs), "{ours}");
        let card = theirs
            .lines()
            .filter(|l| l.starts_with("| A100 SXM4 80 GB |"));
        let on_host: Vec<String> = card
            .map(|row| {
                let mut cells: Vec<String> = row.split(" | ").map(String::from).collect();
                cells[1] = format!("{}, on this host", cells[1]);
                cells[6] = format!("10.3 ms + {restore} µs/token");
                cells.join(" | ")
            })
            .collect();
        assert_eq!(ours[theirs.len()..].lines().collect::<Vec<_>>(), on_host);
    }
    // E0's GPU hits and recomputes are as without a host, the box's host time; its restores slower.
    let e0 = |text: &str| -> Vec<Vec<String>> {
        let rows = section(text, "E0's cells").lines();
        rows.filter(|l| l.starts_with("| ") && l.as_bytes()[2].is_ascii_digit())
            .map(|l| l.split(" | ").map(String::from).collect())
            .collect()
    };
    let (ours, theirs) = (e0(&report), e0(&golden));
    assert_eq!(ours.len(), 7);
    // A forecast's central value, in ms.
    let central = |cell: &str| -> f64 {
        let ms = cell.split(' ').next().unwrap();
        ms.replace(',', "").parse().unwrap()
    };
    for (o, t) in ours.iter().zip(&theirs) {
        assert_eq!((&o[..4], &o[5..]), (&t[..4], &t[5..]));
        assert!(central(&o[4]) > central(&t[4]), "{o:?}");
    }
    // The box's own scenario cells are as they were; the card's say where it is.
    let rows = |text: &str, card: &str| -> Vec<String> {
        let label = format!(" | {card} | ");
        text.lines()
            .filter(|l| l.contains(&label))
            .map(String::from)
            .collect()
    };
    let measured = "RTX 5060 Ti 16 GB (measured)";
    assert_eq!(rows(&report, measured), rows(&golden, measured));
    assert_eq!(
        rows(&report, "A100 SXM4 80 GB (central, on this host)").len(),
        rows(&golden, "A100 SXM4 80 GB (central)").len()
    );
    assert_eq!(
        rows(&report, "A100 SXM4 80 GB (the E1 map's, on this host)").len(),
        1
    );
    assert!(!report.contains("A100 SXM4 80 GB (central) |"), "{report}");
}

/// The idle service is v0's: with the W0 settings it gives the subset run v0 published, 559 requests
/// and 736,878 prompt tokens, 72.4 / 13.3 / 14.3% from the GPU pool, the CPU tier and computation.
#[test]
fn the_idle_service_reproduces_v0_on_the_subset() {
    let cfg = RunConfig {
        service: Service::Idle,
        ..RunConfig::frozen_box(ReplayConfig::default())
    };
    let capacity = TierCapacity::frozen_box(&cfg.geometry);
    let results = run::run(&mut corpus(), &cfg, &[Arm::LruCpu], &[capacity], |_| {}).unwrap();
    let s = &results[0].summary;
    assert_eq!((s.requests, s.prompt_tokens), (559, 736_878));
    assert_eq!(
        (s.gpu_tokens, s.cpu_tokens, s.computed_tokens),
        (533_168, 98_352, 105_358)
    );
    let percent = |x: f64| (1_000.0 * x).round() / 10.0;
    assert_eq!(
        (
            percent(s.shares.gpu),
            percent(s.shares.cpu),
            percent(s.shares.computed)
        ),
        (72.4, 13.3, 14.3)
    );
}

/// `replay-log`, the calibration's instrument, on a run whose engine is known: the subset's
/// simulation, written as AIPerf's export with each request's start and end. Every trace plays once
/// from its start, so a session tree is a trace, and the lines follow the simulation's admissions,
/// which `replay-log` takes in line order at equal times. Every request finds the sources the
/// simulation gave it.
#[test]
fn replay_log_finds_a_simulated_runs_sources_again() {
    let replay = ReplayConfig {
        start_max_ratio: 0.0,
        max_sessions: Some(16),
        duration_s: 1e8,
        grace_s: 0.0,
        ..ReplayConfig::default()
    };
    let cfg = RunConfig::frozen_box(replay.clone());
    let capacity = TierCapacity::frozen_box(&cfg.geometry);
    let engine = || Engine::new(&cfg.engine_config(Arm::LruCpu, capacity), None);
    let mut corpus = corpus();
    let out = Replay::new(
        replay,
        &mut corpus,
        engine(),
        cfg.cost.clone(),
        Service::Idle,
        Retention::None,
        true,
    )
    .run()
    .unwrap();
    let timeline = out.timeline.expect("recorded");
    assert!(out.records.len() > 400, "{} requests", out.records.len());
    let mut log = String::new();
    let mut simulated = Vec::new();
    for event in &timeline.events {
        let TimelineEvent::Admit(i) = *event else {
            continue;
        };
        let r = &out.records[timeline.requests[i]
            .record
            .expect("every request completes")];
        let plan = corpus.plan(r.trace as usize).unwrap();
        let line = json!({
            "metadata": {
                "conversation_id": plan.convs[r.conv as usize].id,
                "turn_index": r.turn,
                "source_trace_id": plan.trace_id,
                "root_correlation_id": plan.trace_id,
                "benchmark_phase": "profiling",
                "request_start_ns": r.issued_ns,
                "request_end_ns": r.done_ns,
            },
            "metrics": {
                "usage_prompt_tokens": {"value": r.prompt_tokens},
                "usage_prompt_cache_read_tokens": {"value": r.sources.gpu + r.sources.cpu},
                "output_sequence_length": {"value": r.output_tokens},
                "time_to_first_token": {"value": r.ttft_ns as f64 / 1e6},
            },
        });
        log += &format!("{line}\n");
        simulated.push(r.sources);
    }
    assert_eq!(simulated.len(), out.records.len());
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("simulated-profile-export.jsonl");
    std::fs::write(&path, log).unwrap();
    let replayed =
        replay_log::replay(&mut corpus, &path, engine(), cfg.replay.block_tokens).unwrap();
    let sources: Vec<PromptSources> = replayed.requests.iter().map(|r| r.sources).collect();
    assert_eq!(sources, simulated);
    assert_eq!(replayed.profiling_requests as usize, simulated.len());
    assert_eq!(replayed.mean_abs_cached_share_error, 0.0);
}

/// One record of W0's AIPerf export (`testdata/aiperf-0.13.0-record.jsonl`), on a trace of the
/// subset: `replay-log` reads its identity, its engine counts and its timing, and takes its prompt
/// from the corpus.
#[test]
fn replay_log_reads_a_record_of_aiperfs_export() {
    let mut corpus = corpus();
    let cfg = RunConfig::frozen_box(ReplayConfig::default());
    let capacity = TierCapacity::frozen_box(&cfg.geometry);
    let engine = Engine::new(&cfg.engine_config(Arm::LruCpu, capacity), None);
    let log = testdata("aiperf-0.13.0-record.jsonl");
    let replayed = replay_log::replay(&mut corpus, &log, engine, cfg.replay.block_tokens).unwrap();
    let [r] = &replayed.requests[..] else {
        panic!("one request: {:?}", replayed.requests)
    };
    assert_eq!(
        (r.conversation.as_str(), r.turn, r.warmup),
        ("04dba6fe621301a1d01fd63b8d02c9645c48", 5, false)
    );
    assert_eq!(
        (
            r.measured_prompt_tokens,
            r.measured_cached_tokens,
            r.output_tokens
        ),
        (2_397, 2_384, 20)
    );
    assert_eq!(r.measured_ttft_ms, 512.077_520_999_999_9);
    // The corpus's prompt, without the engine's chat template, all computed on an empty engine.
    assert_eq!(r.prompt_tokens, 2_352);
    assert_eq!(
        (r.sources.gpu, r.sources.cpu, r.sources.computed),
        (0, 0, 2_352)
    );
}
