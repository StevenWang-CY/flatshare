//! The forecast for a run on another GPU (`lh-sim forecast`): the card's costs as a range, E0's
//! cells, the engine probes' cases, the time-equivalent step cap, and E2's scenarios on the regime
//! map, as one Markdown report.
//!
//! Every number for the other card is modeled from the box's measured costs ([`hardware`]); the
//! box's own measured values stand beside them. Where the box has a measurement of the same thing
//! (E0's cells, the probes on Gray), the forecast is that measurement times the modeled ratio of the
//! two cards, so that what the box's model misses of its own measurement does not carry over.

use std::fmt::Write as _;
use std::path::Path;

use lh_core::{PromptSources, StepCost, StepLoad, TierCapacity, TierCost};

use crate::aiperf::PlanOptions;
use crate::hardware::{
    self, Gpu, Model, QWEN3_4B_FP8, QWEN3_14B_FP8, RTX_5060_TI, Scaling, TOKENS_PER_CALL,
};
use crate::metrics::Summary;
use crate::regime::{self, thousands};
use crate::replay::{ReplayConfig, Service};
use crate::run::{self, Arm, RunConfig};
use crate::scheduler::StepBudget;

/// The settings that bound a forecast ([`Scaling::LOW`], [`Scaling::central`], [`Scaling::high`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Low,
    Central,
    High,
}

impl Variant {
    pub const ALL: [Variant; 3] = [Variant::Low, Variant::Central, Variant::High];

    pub fn scaling(self, gpu: &Gpu) -> Scaling {
        match self {
            Variant::Low => Scaling::LOW,
            Variant::Central => Scaling::central(gpu),
            Variant::High => Scaling::high(gpu),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Variant::Low => "low",
            Variant::Central => "central",
            Variant::High => "high",
        }
    }
}

/// A card's step and tier costs for a model.
#[derive(Clone, Debug, PartialEq)]
pub struct Costs {
    pub steps: StepCost,
    pub tier: TierCost,
}

impl Costs {
    /// The box's measured costs for its own model; otherwise `gpu`'s for `model` under `scaling`.
    pub fn of(gpu: &Gpu, model: &Model, scaling: Scaling) -> Self {
        if gpu.measured() && *model == QWEN3_4B_FP8 {
            return Self::frozen_box();
        }
        Self {
            steps: gpu.step_cost_with(model, scaling),
            tier: gpu.tier_cost_with(model, scaling),
        }
    }

    pub fn frozen_box() -> Self {
        Self {
            steps: StepCost::frozen_box(),
            tier: TierCost::frozen_box(),
        }
    }
}

/// E0's prompt lengths.
pub const E0_LENGTHS: [u64; 7] = [1_024, 2_048, 4_096, 8_192, 16_384, 24_576, 31_744];

/// E0's p50 cells on the box, 2026-10-22, in ms (lh-core's tests hold the same): a GPU hit, a
/// CPU-tier restore and a recompute, at each of [`E0_LENGTHS`]. Measured.
pub const E0_BOX_P50: [[f64; 7]; 3] = [
    [45.0, 40.0, 48.0, 64.0, 97.0, 128.0, 156.0],
    [64.0, 95.0, 157.0, 281.0, 509.0, 722.0, 903.0],
    [135.0, 198.0, 437.0, 1_114.0, 3_264.0, 6_506.0, 10_239.0],
];

/// The modeled time to first token of E0's three requests of `tokens` tokens (a GPU hit, a restore
/// and a recompute, as `[hit, restore, cold]`): each alone on the engine, with E0's host time, its
/// last block computed.
pub fn e0_model(costs: &Costs, tokens: u64) -> [f64; 3] {
    let steps = StepCost {
        host: StepCost::frozen_box().e0_host(&TierCost::frozen_box()),
        ..costs.steps.clone()
    };
    let block = costs.tier.block_tokens;
    let sources = [
        PromptSources {
            gpu: tokens - block,
            cpu: 0,
            computed: block,
        },
        PromptSources {
            gpu: 0,
            cpu: tokens - block,
            computed: block,
        },
        PromptSources {
            gpu: 0,
            cpu: 0,
            computed: tokens,
        },
    ];
    sources.map(|s| steps.alone_ttft_ms(&costs.tier, s, StepBudget::FROZEN_BOX.max_batched_tokens))
}

/// The KV case on Gray, the second RTX 5060 Ti with the box's units (`docs/BENCHMARKS.md`, "The KV
/// case: make room"), measured: stock vLLM's priority-0 time to first token in its three runs,
/// make-room's with the KV preemption and the budget cap in its two, and either's on an idle pool.
pub const KV_STOCK_MS: [f64; 3] = [4_923.0, 5_069.0, 5_077.0];
pub const KV_MAKE_ROOM_MS: [f64; 2] = [225.0, 297.0];
pub const KV_IDLE_MS: [f64; 2] = [107.0, 137.0];

/// The two priority-2 requests' context in the KV case (prompts of 17,999 tokens, decoding), and the
/// voice turn's prompt (about 750 tokens).
const KV_CONTEXT: u64 = 18_000;
const VOICE_PROMPT: u64 = 750;

/// The prefill case on Gray, measured: the reply's tokens after its first came 62 to 73 ms apart
/// behind the agents' 512-token chunks on stock vLLM, and 16 to 20 ms apart with the budget cap
/// holding them; the medians' first and 16th tokens give these.
pub const PREFILL_GAP_STOCK_MS: f64 = (1_395.0 - 361.0) / 15.0;
pub const PREFILL_GAP_CAP_MS: f64 = (573.0 - 320.0) / 15.0;

/// The agents' context in the prefill case: halfway through their 10,000-token prompts.
const PREFILL_CONTEXT: u64 = 5_000;

/// The median request's context in E2's discriminating scenario and in the whole corpus
/// (`docs/EXPERIMENTS.md`, E2): where a time-equivalent step cap is set.
pub const MEDIAN_CONTEXTS: [(&str, u64); 2] = [
    ("the discriminating scenario's median request", 13_900),
    ("the whole corpus's", 7_900),
];

/// A step of two decodes at the KV case's context.
fn two_decodes_ms(steps: &StepCost) -> f64 {
    let mut load = StepLoad::default();
    load.add_decode(KV_CONTEXT);
    load.add_decode(KV_CONTEXT);
    steps.step_ms(&load)
}

/// A step of the voice reply's decode, with an agent chunk of `chunk` tokens beside it or not.
fn reply_step_ms(steps: &StepCost, chunk: u64) -> f64 {
    let mut load = StepLoad::default();
    if chunk > 0 {
        load.add_prefill(chunk, PREFILL_CONTEXT);
    }
    load.add_decode(VOICE_PROMPT);
    steps.step_ms(&load)
}

/// A step carrying one `tokens`-token prefill chunk after `context` cached tokens, alone.
pub fn chunk_ms(steps: &StepCost, tokens: u64, context: u64) -> f64 {
    let mut load = StepLoad::default();
    load.add_prefill(tokens, context);
    steps.step_ms(&load)
}

/// The largest step cap, a multiple of 256 tokens, whose chunk after `context` tokens takes `steps`
/// no longer than the box's 512-token chunk takes the box: a realtime decode then waits no longer
/// behind a chunk than on the box. `None` when even 256 tokens take longer.
pub fn time_equivalent_cap(steps: &StepCost, context: u64) -> Option<u64> {
    let budget = StepBudget::FROZEN_BOX.max_batched_tokens;
    let target = chunk_ms(&StepCost::frozen_box(), budget, context);
    (1..=64)
        .map(|i| i * 256)
        .take_while(|&b| chunk_ms(steps, b, context) <= target)
        .last()
}

/// The probes' cases on `costs`, against the box's model: (stock's wait in the KV case, make-room's
/// first token there, the reply's gap behind agent chunks, its gap with the budget cap), each as a
/// range in ms, the box's measurement times the modeled ratio.
pub fn probes(costs: &Costs) -> [(f64, f64); 4] {
    let (b, g) = (&StepCost::frozen_box(), &costs.steps);
    let decode = two_decodes_ms(g) / two_decodes_ms(b);
    let budget = StepBudget::FROZEN_BOX.max_batched_tokens;
    let saved = b.prefill_ms(VOICE_PROMPT, 0, budget) - g.prefill_ms(VOICE_PROMPT, 0, budget);
    let stock = (KV_STOCK_MS[0] * decode, KV_STOCK_MS[2] * decode);
    // Each run's first token: the voice turn's prefill, faster by what the card saves, and the
    // excess over the idle pool's (make-room's preemption and the frees it waits for, steps of the
    // full pool) by the decode ratio.
    let idle = (KV_IDLE_MS[0] + KV_IDLE_MS[1]) / 2.0;
    let run = |x: f64| x - saved - (1.0 - decode) * (x - idle);
    let make_room = (run(KV_MAKE_ROOM_MS[0]), run(KV_MAKE_ROOM_MS[1]));
    let gap = |chunk: u64, measured: f64| {
        let r = reply_step_ms(g, chunk) / reply_step_ms(b, chunk);
        (measured * r, measured * r)
    };
    [
        stock,
        make_room,
        gap(budget - 1, PREFILL_GAP_STOCK_MS),
        gap(0, PREFILL_GAP_CAP_MS),
    ]
}

/// One cell of the forecast's map: a scenario on a card, with one setting's costs.
#[derive(Clone, Debug)]
pub struct Cell {
    pub scenario: &'static str,
    /// Whether the cell replays the discriminating corpus rather than the whole one.
    pub long: bool,
    /// The card and setting, as the table names them.
    pub label: String,
    pub model: Model,
    pub costs: Costs,
    pub sessions: usize,
    pub pool_tokens: u64,
    pub cpu_bytes: u64,
}

impl Cell {
    fn capacity(&self) -> TierCapacity {
        let g = &self.model.geometry;
        TierCapacity {
            gpu_blocks: self.pool_tokens / g.block_tokens,
            cpu_blocks: g.blocks_in_bytes(self.cpu_bytes),
        }
    }
}

/// E2's scenarios on `gpu`, as the A100 campaign's design names them, with the box's own cells
/// beside the core's and the calibration's.
pub fn cells(gpu: &Gpu) -> Vec<Cell> {
    let pool = lh_core::FROZEN_BOX_POOL_TOKENS;
    let cpu = lh_core::FROZEN_BOX_CPU_BYTES;
    let central = Variant::Central.scaling(gpu);
    let cell = |scenario, long, label: String, model: Model, costs, sessions, pool_tokens: u64| {
        // The CPU tier is 4x the pool, as the box's 20 GiB is of its 5 GiB.
        let cpu_bytes = if pool_tokens == pool {
            cpu
        } else {
            4 * model.geometry.bytes_for_tokens(pool_tokens)
        };
        Cell {
            scenario,
            long,
            label,
            model,
            costs,
            sessions,
            pool_tokens,
            cpu_bytes,
        }
    };
    let name = |s: &str| format!("{} ({s})", gpu.title);
    let measured = format!("{} (measured)", RTX_5060_TI.title);
    let gpu4 = Costs::of(gpu, &QWEN3_4B_FP8, central);
    let gpu14 = Costs::of(gpu, &QWEN3_14B_FP8, central);
    let own4 = gpu.model_pool_tokens(&QWEN3_4B_FP8);
    let own14 = gpu.model_pool_tokens(&QWEN3_14B_FP8);
    let core = "A: the core, the box's pool";
    let calibration = "the calibration point, the whole corpus";
    let mut out = vec![
        cell(
            core,
            true,
            measured.clone(),
            QWEN3_4B_FP8,
            Costs::frozen_box(),
            24,
            pool,
        ),
        cell(
            core,
            true,
            name("the E1 map's"),
            QWEN3_4B_FP8,
            Costs::of(gpu, &QWEN3_4B_FP8, Scaling::MAP),
            24,
            pool,
        ),
        cell(
            core,
            true,
            name("central"),
            QWEN3_4B_FP8,
            gpu4.clone(),
            24,
            pool,
        ),
        cell(
            calibration,
            false,
            measured,
            QWEN3_4B_FP8,
            Costs::frozen_box(),
            16,
            pool,
        ),
        cell(
            calibration,
            false,
            name("central"),
            QWEN3_4B_FP8,
            gpu4.clone(),
            16,
            pool,
        ),
    ];
    out.push(cell(
        "A': the box's pool, more agents",
        true,
        name("central"),
        QWEN3_4B_FP8,
        gpu4.clone(),
        32,
        pool,
    ));
    if own14 > 0 {
        for sessions in [24, 32, 40, 48] {
            let c = cell(
                "C': Qwen3-14B-FP8 at its own pool",
                true,
                name("central"),
                QWEN3_14B_FP8,
                gpu14.clone(),
                sessions,
                own14,
            );
            out.push(c);
        }
    }
    out.push(cell(
        "B: twice the box's pool and sessions",
        true,
        name("central"),
        QWEN3_4B_FP8,
        gpu4.clone(),
        48,
        2 * pool,
    ));
    out.push(cell(
        "D: Qwen3-4B at the card's own pool",
        true,
        name("central"),
        QWEN3_4B_FP8,
        gpu4,
        24,
        own4,
    ));
    out
}

/// What one cell ran to, the means over its seeds.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    pub pressure_mean: f64,
    pub pressure_peak: f64,
    pub pi: f64,
    pub rho: f64,
    /// The time running agent requests leave a realtime tier of two calls short of KV, and the time
    /// the agents' working set does.
    pub running_share: f64,
    pub working_set_share: f64,
    /// The card's step time over the replay's.
    pub busy: f64,
    /// The requests that waited for free blocks.
    pub stalled: f64,
    pub computed: f64,
    pub resume_p50_ms: f64,
    pub resume_p90_ms: f64,
}

impl Placed {
    fn of(summaries: &[(Summary, f64)]) -> Self {
        let n = summaries.len().max(1) as f64;
        let mean = |f: &dyn Fn(&Summary, f64) -> f64| {
            summaries.iter().map(|(s, w)| f(s, *w)).sum::<f64>() / n
        };
        Self {
            pressure_mean: mean(&|s, _| s.regime.kv_pressure_mean),
            pressure_peak: mean(&|s, _| s.regime.kv_pressure_peak),
            pi: mean(&|s, _| s.regime.pi),
            rho: mean(&|s, _| s.regime.rho),
            running_share: mean(&|s, _| s.replay.realtime_blocked_share.unwrap_or(0.0)),
            working_set_share: mean(&|s, _| s.replay.realtime_short_share.unwrap_or(0.0)),
            busy: mean(&|s, window| {
                s.engine.step_ns as f64 / 1e9 / (window + s.replay.profiling_start_s)
            }),
            stalled: mean(&|s, _| s.stalls as f64 / s.requests.max(1) as f64),
            computed: mean(&|s, _| s.shares.computed),
            resume_p50_ms: mean(&|s, _| s.resume_ttft_ms.p50),
            resume_p90_ms: mean(&|s, _| s.resume_ttft_ms.p90),
        }
    }
}

/// The replay every cell runs: E2's agent replay as lh-sim models it (600 s and 60 s of grace,
/// trace pauses), a realtime tier of two calls.
pub fn replay(base: &ReplayConfig) -> ReplayConfig {
    ReplayConfig {
        realtime_tokens: 2 * TOKENS_PER_CALL,
        ..base.clone()
    }
}

/// Runs `lru-cpu`, the stock engine's policy, in every cell at every seed on `jobs` threads, and
/// returns each cell's means.
pub fn place(
    whole: &Path,
    long: &Path,
    plan: PlanOptions,
    base: &ReplayConfig,
    cells: &[Cell],
    seeds: &[u64],
    jobs: usize,
) -> Result<Vec<Placed>, String> {
    let mut runs: Vec<Vec<(Summary, f64)>> = vec![Vec::new(); cells.len()];
    for long_corpus in [false, true] {
        let items: Vec<(usize, u64)> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.long == long_corpus)
            .flat_map(|(i, _)| seeds.iter().map(move |&s| (i, s)))
            .collect();
        if items.is_empty() {
            continue;
        }
        let traces = if long_corpus { long } else { whole };
        let done = regime::par_map(traces, plan, &items, jobs, |corpus, &(i, seed)| {
            let cell = &cells[i];
            let cfg = RunConfig {
                replay: ReplayConfig {
                    concurrency: cell.sessions,
                    seed,
                    ..replay(base)
                },
                cost: cell.costs.tier.clone(),
                service: Service::Contention {
                    budget: StepBudget::FROZEN_BOX,
                    steps: cell.costs.steps.clone(),
                },
                geometry: cell.model.geometry,
                reference: Arm::LruCpu,
            };
            let window = cfg.replay.duration_s + cfg.replay.grace_s;
            let (result, _) = run::run_arm(corpus, &cfg, Arm::LruCpu, cell.capacity(), false)
                .map_err(|e| format!("{} ({}): {e}", cell.scenario, cell.label))?;
            Ok((i, result.summary, window))
        })?;
        for (i, summary, window) in done {
            runs[i].push((summary, window));
        }
    }
    Ok(runs.iter().map(|r| Placed::of(r)).collect())
}

fn ms(x: f64) -> String {
    if x >= 100.0 {
        thousands(x.round() as u64)
    } else {
        format!("{x:.1}")
    }
}

fn range(lo: f64, hi: f64) -> String {
    let (a, b) = (ms(lo.min(hi)), ms(lo.max(hi)));
    if a == b { a } else { format!("{a}–{b}") }
}

fn pct(x: f64) -> String {
    format!("{:.0}%", 100.0 * x)
}

/// The costs table: the box's, and `gpu`'s in each setting for `model`.
pub fn costs_table(gpu: &Gpu, model: &Model) -> String {
    let mut text = String::from(
        "| Card | Costs | Step: fixed | per token | per pair | per KV token read | Restore \
         | Cold prefill of 31,744 tokens |\n|---|---|---:|---:|---:|---:|---|---:|\n",
    );
    let mut row = |card: &str, label: &str, c: &Costs| {
        let s = &c.steps;
        writeln!(
            text,
            "| {card} | {label} | {:.2} ms | {:.2} µs | {:.2} ns | {:.1} ns \
             | {:.1} ms + {:.2} µs/token | {} ms |",
            s.step_ms,
            s.us_per_token,
            s.ns_per_pair,
            s.ns_per_context,
            c.tier.cpu_restore.ms,
            c.tier.cpu_restore.us_per_token,
            thousands(c.tier.recompute_ms(31_744).round() as u64),
        )
        .expect("writing to a String");
    };
    if *model == QWEN3_4B_FP8 {
        row(RTX_5060_TI.title, "measured", &Costs::frozen_box());
    }
    row(
        gpu.title,
        "the E1 map's",
        &Costs::of(gpu, model, Scaling::MAP),
    );
    for v in Variant::ALL {
        row(gpu.title, v.name(), &Costs::of(gpu, model, v.scaling(gpu)));
    }
    text
}

/// E0's cells: the box's measured p50s and `gpu`'s forecast, central with the low–high range.
pub fn e0_table(gpu: &Gpu) -> String {
    let mut text = String::from(
        "| Prompt tokens | GPU hit: box | forecast | CPU restore: box | forecast | Recompute: box \
         | forecast |\n|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    let box_costs = Costs::frozen_box();
    let costs = Variant::ALL.map(|v| Costs::of(gpu, &QWEN3_4B_FP8, v.scaling(gpu)));
    for (i, &tokens) in E0_LENGTHS.iter().enumerate() {
        let base = e0_model(&box_costs, tokens);
        let models = costs.each_ref().map(|c| e0_model(c, tokens));
        write!(text, "| {} |", thousands(tokens)).expect("writing to a String");
        for kind in 0..3 {
            let measured = E0_BOX_P50[kind][i];
            let f = |v: usize| measured * models[v][kind] / base[kind];
            let (lo, hi) = (f(0).min(f(2)), f(0).max(f(2)));
            write!(
                text,
                " {} | {} ({}) |",
                ms(measured),
                ms(f(1)),
                range(lo, hi)
            )
            .expect("writing to a String");
        }
        text.push('\n');
    }
    text
}

/// The probes, the time-equivalent caps and their sources, as Markdown bullets.
pub fn probes_section(gpu: &Gpu) -> String {
    let costs = Variant::ALL.map(|v| Costs::of(gpu, &QWEN3_4B_FP8, v.scaling(gpu)));
    let p = costs.each_ref().map(probes);
    let span = |k: usize| {
        let lo = p.iter().map(|x| x[k].0).fold(f64::INFINITY, f64::min);
        let hi = p.iter().map(|x| x[k].1).fold(0.0, f64::max);
        (range(p[1][k].0, p[1][k].1), range(lo, hi))
    };
    let mut text = String::new();
    let (stock, stock_all) = span(0);
    let (make_room, make_room_all) = span(1);
    let box_decode = two_decodes_ms(&StepCost::frozen_box());
    writeln!(
        text,
        "- **The KV case** (`probe.py kv`). Stock vLLM's voice turn waits while two priority-2 \
         requests decode into the pool's last free blocks: on Gray {}–{} ms, which the box's model \
         brackets at 176–192 steps of {box_decode:.1} ms ({}–{} ms). On the {}: {stock} ms \
         (low–high {stock_all}). make-room's first token, 225 and 297 ms on Gray against 107–137 ms \
         on an idle pool: {make_room} ms ({make_room_all}).",
        thousands(KV_STOCK_MS[0] as u64),
        thousands(KV_STOCK_MS[2] as u64),
        thousands((176.0 * box_decode).round() as u64),
        thousands((192.0 * box_decode).round() as u64),
        gpu.title,
    )
    .expect("writing to a String");
    let (gap_stock, gap_stock_all) = span(2);
    let (gap_cap, gap_cap_all) = span(3);
    writeln!(
        text,
        "- **The prefill case** (`probe.py prefill`). The reply's tokens come {PREFILL_GAP_STOCK_MS:.0} \
         ms apart behind the agents' chunks on stock vLLM and {PREFILL_GAP_CAP_MS:.0} ms apart with \
         the budget cap holding them, on Gray (the medians' 1st to 16th tokens). On the {}: \
         {gap_stock} ms ({gap_stock_all}) and {gap_cap} ms ({gap_cap_all}), with the chunks at \
         {} tokens of context.",
        gpu.title,
        thousands(PREFILL_CONTEXT),
    )
    .expect("writing to a String");
    for (what, context) in MEDIAN_CONTEXTS {
        let box_chunk = chunk_ms(
            &StepCost::frozen_box(),
            StepBudget::FROZEN_BOX.max_batched_tokens,
            context,
        );
        let caps: Vec<String> = Variant::ALL
            .iter()
            .zip(&costs)
            .map(|(v, c)| match time_equivalent_cap(&c.steps, context) {
                Some(b) => format!(
                    "{} tokens ({}, {:.0} ms)",
                    thousands(b),
                    v.name(),
                    chunk_ms(&c.steps, b, context)
                ),
                None => format!("none ({})", v.name()),
            })
            .collect();
        writeln!(
            text,
            "- **The time-equivalent step cap** at {what} context ({} tokens), where the box's \
             512-token chunk takes {box_chunk:.0} ms: {}. The long-prefill threshold is half the cap, \
             as the box's 256 is of its 512.",
            thousands(context),
            caps.join("; "),
        )
        .expect("writing to a String");
    }
    text
}

/// The placements table.
pub fn placements_table(cells: &[Cell], placed: &[Placed], seeds: &[u64]) -> String {
    let seeds: Vec<_> = seeds.iter().map(u64::to_string).collect();
    let mut text = format!(
        "Means over seeds {}; `lru-cpu`, the stock engine's policy; a realtime tier of two calls. \
         Running / working set: the time running agent requests / the agents' working set leave \
         those calls short of KV. Busy: the card's step time over the replay's. Stalled: the \
         requests that waited for free blocks.\n\n\
         | Scenario | Card | Sessions | Pool | CPU tier | Pressure | π | ρ | Running / working set \
         | Busy | Stalled | Computed | Resume TTFT p50 / p90 |\n\
         |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        seeds.join(", "),
    );
    for (c, p) in cells.iter().zip(placed) {
        writeln!(
            text,
            "| {} | {} | {} | {} tokens | {} GiB | {:.1} / {:.1} | {:.1} | {:.2} | {} / {} | {} | {} \
             | {} | {} / {} ms |",
            c.scenario,
            c.label,
            c.sessions,
            thousands(c.pool_tokens),
            thousands((c.cpu_bytes as f64 / (1u64 << 30) as f64).round() as u64),
            p.pressure_mean,
            p.pressure_peak,
            p.pi,
            p.rho,
            pct(p.running_share),
            pct(p.working_set_share),
            pct(p.busy),
            pct(p.stalled),
            pct(p.computed),
            thousands(p.resume_p50_ms.round() as u64),
            thousands(p.resume_p90_ms.round() as u64),
        )
        .expect("writing to a String");
    }
    text
}

/// What the numbers say the run will show, as sentences built from them.
pub fn reading(gpu: &Gpu, cells: &[Cell], placed: &[Placed]) -> String {
    let all = |scenario: &str| {
        cells
            .iter()
            .zip(placed)
            .filter(|(c, _)| c.scenario.starts_with(scenario) && c.label.contains("central"))
            .collect::<Vec<_>>()
    };
    let one = |scenario: &str, label: &str| {
        cells
            .iter()
            .zip(placed)
            .find(|(c, _)| c.scenario.starts_with(scenario) && c.label.contains(label))
            .map(|(_, p)| p)
    };
    let row = |p: &Placed| {
        format!(
            "pressure {:.1}, running share {}, {} busy, {} of requests stalled",
            p.pressure_mean,
            pct(p.running_share),
            pct(p.busy),
            pct(p.stalled)
        )
    };
    let mut text = String::new();
    let central = Costs::of(gpu, &QWEN3_4B_FP8, Scaling::central(gpu));
    let decode = two_decodes_ms(&central.steps) / two_decodes_ms(&StepCost::frozen_box());
    if let (Some(b), Some(g)) = (one("A:", "measured"), one("A:", "central")) {
        writeln!(
            text,
            "- **The core, at the box's pool and 24 sessions.** On the {}: {}; on the box: {}. \
             Running agent requests leave two calls short of KV {} of the time against the box's {}, \
             and such a wait, a decode-bound stretch, runs at {:.2} of the box's step time: the \
             KV-blocking regime the change addresses, with shorter waits.",
            gpu.title,
            row(g),
            row(b),
            pct(g.running_share),
            pct(b.running_share),
            decode,
        )
        .expect("writing to a String");
    }
    let named = |what: &str, scenario: &str| {
        let cells = all(scenario);
        if cells.is_empty() {
            return None;
        }
        let parts: Vec<String> = cells
            .iter()
            .map(|(c, p)| format!("{} sessions: {}", c.sessions, row(p)))
            .collect();
        Some(format!("- **{what}.** {}.", parts.join("; ")))
    };
    for (what, scenario) in [
        ("More agents at the box's pool", "A'"),
        ("Qwen3-14B-FP8 at the card's own pool", "C'"),
        ("Twice the box's pool and sessions", "B:"),
        ("Qwen3-4B at the card's own pool", "D:"),
    ] {
        if let Some(line) = named(what, scenario) {
            writeln!(text, "{line}").expect("writing to a String");
        }
    }
    // Which of the card's cells leaves the calls short of KV the most: where the change has the
    // most to do.
    let modeled = cells
        .iter()
        .zip(placed)
        .filter(|(c, _)| c.label.contains("central") && c.long);
    let on_box = one("A:", "measured");
    if let Some((c, p)) = modeled.max_by(|a, b| a.1.running_share.total_cmp(&b.1.running_share)) {
        writeln!(
            text,
            "- **The most time short of KV** among the {}'s scenarios: {}, {} sessions ({}{}).",
            gpu.title,
            c.scenario,
            c.sessions,
            pct(p.running_share),
            on_box.map_or(String::new(), |b| format!(
                ", against the box's {} in the core",
                pct(b.running_share)
            )),
        )
        .expect("writing to a String");
    }
    text
}

/// The whole report for `gpu`.
pub fn report(gpu: &Gpu, cells: &[Cell], placed: &[Placed], seeds: &[u64]) -> String {
    let mut text = String::new();
    writeln!(
        text,
        "# Forecast: the {} against the frozen box\n\n\
         Written by `lh-sim forecast` (`make forecast-a100`) from the box's measured costs, the \
         card's published peaks and the path vLLM 0.32.0 takes for the box's checkpoint on it ({}; \
         `crates/lh-sim/src/hardware.rs`). Every number for the {} is modeled; the box's are \
         measured where they say so. Three settings bound it: **low**, the peaks' ratios alone with \
         Marlin as fast as a BF16 GEMM; **central**, with the box's 1.7 ms of step overhead kept, \
         Marlin {:.2}× a BF16 GEMM (vLLM PR #61224's L4) and FlashAttention 2 at its best published \
         efficiency on the card; **high**, with FlashAttention 2 at its worst published. The E1 \
         map's model, on which `lh-sim hardware` places every card, scales the fixed step with \
         bandwidth and keeps the box's attention efficiency.\n",
        gpu.title,
        gpu.linear.label(),
        gpu.title,
        hardware::MARLIN_FACTOR,
    )
    .expect("writing to a String");
    writeln!(text, "## Step and tier costs, Qwen3-4B-Instruct-2507-FP8\n")
        .expect("writing to a String");
    text.push_str(&costs_table(gpu, &QWEN3_4B_FP8));
    if gpu.model_pool_tokens(&QWEN3_14B_FP8) > 0 {
        writeln!(text, "\n## Step and tier costs, Qwen3-14B-FP8\n").expect("writing to a String");
        text.push_str(&costs_table(gpu, &QWEN3_14B_FP8));
    }
    writeln!(
        text,
        "\n## E0's cells\n\nTime to first token of a request alone, in ms: the box's measured p50 \
         (2026-10-22), and the forecast as that times the modeled ratio of the two cards, central \
         (low–high).\n"
    )
    .expect("writing to a String");
    text.push_str(&e0_table(gpu));
    writeln!(text, "\n## The engine probes\n").expect("writing to a String");
    text.push_str(&probes_section(gpu));
    writeln!(
        text,
        "\n## E2's scenarios on the regime map\n\nEach cell replays E2's agents in `lh-sim`'s model \
         of vLLM 0.32.0's two-tier KV cache and step loop with the card's central costs; the \
         discriminating scenario replays `weka-x32-long`, the calibration point the whole 32x \
         corpus.\n"
    )
    .expect("writing to a String");
    text.push_str(&placements_table(cells, placed, seeds));
    writeln!(text, "\n## What the forecast says the run will show\n").expect("writing to a String");
    text.push_str(&reading(gpu, cells, placed));
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{A100_40GB, L4};

    #[test]
    fn the_box_forecasts_its_own_measurements() {
        let b = Costs::frozen_box();
        assert_eq!(
            Costs::of(&RTX_5060_TI, &QWEN3_4B_FP8, Scaling::central(&RTX_5060_TI)),
            b
        );
        // The probes' forecast on the box is the box's measurement.
        let p = probes(&b);
        assert_eq!(p[0], (4_923.0, 5_077.0));
        assert_eq!(p[1], (225.0, 297.0));
        assert!((p[2].0 - PREFILL_GAP_STOCK_MS).abs() < 1e-9);
        assert!((p[3].0 - PREFILL_GAP_CAP_MS).abs() < 1e-9);
        // The box's own step cap is time-equivalent to itself.
        for (_, context) in MEDIAN_CONTEXTS {
            assert_eq!(time_equivalent_cap(&b.steps, context), Some(512));
        }
        // E0's forecast on the box is its measured cells.
        let table = e0_table(&RTX_5060_TI);
        assert!(
            table.contains(
                "| 31,744 | 156 | 156 (156) | 903 | 903 (903) | 10,239 | 10,239 (10,239) |"
            ),
            "{table}"
        );
    }

    #[test]
    fn the_a100s_probes_follow_its_decode_and_chunk_steps() {
        let c = Costs::of(&A100_40GB, &QWEN3_4B_FP8, Scaling::central(&A100_40GB));
        // Two 18k decodes: 9.2 ms against the box's 27.6, so stock waits 1.6-1.7 s.
        assert!(
            (two_decodes_ms(&c.steps) - 9.18).abs() < 0.01,
            "{}",
            two_decodes_ms(&c.steps)
        );
        assert!((two_decodes_ms(&StepCost::frozen_box()) - 27.62).abs() < 0.01);
        let p = probes(&c);
        assert!(
            (1_600.0..1_700.0).contains(&p[0].0) && (1_650.0..1_720.0).contains(&p[0].1),
            "{p:?}"
        );
        // make-room's first token: 225 and 297 ms less 15 ms of prefill and two thirds of their
        // excess over the idle pool's 122 ms.
        assert!(
            (130.0..150.0).contains(&p[1].0) && (155.0..175.0).contains(&p[1].1),
            "{p:?}"
        );
        // The time-equivalent cap: 1,536 at the discriminating scenario's 13.9k, 1,280 at 7.9k.
        assert_eq!(time_equivalent_cap(&c.steps, 13_900), Some(1_536));
        assert_eq!(time_equivalent_cap(&c.steps, 7_900), Some(1_280));
        // A slower card gets a smaller cap: the L4's Marlin layers at 106.6 µs a token.
        let l4 = Costs::of(&L4, &QWEN3_4B_FP8, Scaling::central(&L4));
        assert!(time_equivalent_cap(&l4.steps, 7_900).is_none_or(|b| b < 512));
    }

    #[test]
    fn the_cells_name_the_campaigns_scenarios() {
        let cells = cells(&A100_40GB);
        let names: Vec<_> = cells
            .iter()
            .map(|c| (c.scenario, c.sessions, c.pool_tokens))
            .collect();
        assert_eq!(names[0], ("A: the core, the box's pool", 24, 36_400));
        assert_eq!(cells[0].label, "RTX 5060 Ti 16 GB (measured)");
        assert_eq!(cells[0].cpu_bytes, 20 << 30);
        assert!(names.contains(&("C': Qwen3-14B-FP8 at its own pool", 48, 117_344)));
        assert!(names.contains(&("C': Qwen3-14B-FP8 at its own pool", 32, 117_344)));
        assert!(names.contains(&("A': the box's pool, more agents", 32, 36_400)));
        assert!(names.contains(&("B: twice the box's pool and sessions", 48, 72_800)));
        assert!(names.contains(&("D: Qwen3-4B at the card's own pool", 24, 211_168)));
        // The CPU tier is 4x the pool off the box's pool: the 14B's 18.3 GiB of KV ask 73.3 GiB.
        let c14 = cells.iter().find(|c| c.model == QWEN3_14B_FP8).unwrap();
        assert_eq!(c14.cpu_bytes, 4 * 117_344 * 163_840);
        assert_eq!(c14.capacity().gpu_blocks, 7_334);
        // The core runs the discriminating corpus, the calibration point the whole one.
        assert!(cells[0].long && !cells[3].long);
        // A card too small for the 14B gets no C' cells.
        assert!(
            !super::cells(&RTX_5060_TI)
                .iter()
                .any(|c| c.model == QWEN3_14B_FP8)
        );
    }

    #[test]
    fn the_tables_carry_the_settings() {
        let t = costs_table(&A100_40GB, &QWEN3_4B_FP8);
        assert!(
            t.contains(
                "| RTX 5060 Ti 16 GB | measured | 11.56 ms | 47.99 µs | 14.17 ns | 429.3 ns |"
            ),
            "{t}"
        );
        assert!(
            t.contains("| A100 40 GB | central | 4.55 ms | 41.33 µs | 2.59 ns | 123.7 ns |"),
            "{t}"
        );
        assert!(
            t.contains("| A100 40 GB | low | 3.33 ms | 29.19 µs | 2.15 ns |"),
            "{t}"
        );
        assert!(
            t.contains("| A100 40 GB | high | 4.55 ms | 41.33 µs | 3.78 ns |"),
            "{t}"
        );
        assert!(
            t.contains("| A100 40 GB | the E1 map's | 3.33 ms | 41.33 µs | 2.15 ns |"),
            "{t}"
        );
        let t14 = costs_table(&A100_40GB, &QWEN3_14B_FP8);
        assert!(!t14.contains("measured"), "{t14}");
        let probes = probes_section(&A100_40GB);
        assert!(probes.contains("1,536 tokens (central"), "{probes}");
        assert!(probes.contains("on Gray 4,923–5,077 ms"), "{probes}");
    }
}
