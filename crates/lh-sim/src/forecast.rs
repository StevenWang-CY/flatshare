//! The forecast for a run on another GPU (`lh-sim forecast`): the card's costs as a range, E0's
//! cells, the engine probes' cases, the time-equivalent step cap, and E2's scenarios on the regime
//! map, as one Markdown report.
//!
//! Every number for the other card is modeled from the box's measured costs ([`hardware`]); the
//! box's own measured values stand beside them. Where the box has a measurement of the same thing
//! (E0's cells, the probes on Gray), the forecast is that measurement times the modeled ratio of the
//! two cards, so that what the box's model misses of its own measurement does not carry over.
//! Placed on a measured host ([`Host`]), the card's cells take the API server's time from its CPU
//! and the restores from its link, and the costs tables add the card's costs there. Once the card
//! has a step-time model and E0 tier fits of its own, the forecast places the calibration point and
//! the core with them too ([`fitted_cells`]).

use std::fmt::Write as _;
use std::path::Path;

use lh_core::{PromptSources, StepCost, StepLoad, TierCapacity, TierCost};

use crate::aiperf::PlanOptions;
use crate::hardware::{
    self, Gpu, Host, Model, QWEN3_4B_FP8, QWEN3_14B_FP8, RTX_5060_TI, Scaling, TOKENS_PER_CALL,
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
        Self::on(gpu, model, scaling, None)
    }

    /// The same with the card on `host` ([`Gpu::step_cost_on`]), and as [`of`](Self::of) without
    /// one.
    pub fn on(gpu: &Gpu, model: &Model, scaling: Scaling, host: Option<&Host>) -> Self {
        // Every scaling gives the box its measured step terms for its own model, the E1 map's
        // exactly, so that its own host gives its measured costs back.
        let scaling = if gpu.measured() && *model == QWEN3_4B_FP8 {
            if host.is_none() {
                return Self::frozen_box();
            }
            Scaling::MAP
        } else {
            scaling
        };
        Self {
            steps: gpu.step_cost_on(model, scaling, host),
            tier: gpu.tier_cost_on(model, scaling, host),
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
/// and a recompute, as `[hit, restore, cold]`): each alone on the engine, with E0's host time on
/// `host` (the box's without one), its last block computed.
pub fn e0_model(costs: &Costs, host: Option<&Host>, tokens: u64) -> [f64; 3] {
    let steps = StepCost {
        host: hardware::e0_host(host),
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

/// The scenarios at the box's pool: the core, more agents than its 24 sessions, and the calibration
/// point.
const CORE: &str = "A: the core, the box's pool";
const MORE_AGENTS: &str = "A': the box's pool, more agents";
const CALIBRATION: &str = "the calibration point, the whole corpus";

/// A setting's name, which says so when the card is placed on a host.
fn setting(name: &str, host: Option<&Host>) -> String {
    match host {
        Some(_) => format!("{name}, on this host"),
        None => name.to_string(),
    }
}

/// E2's scenarios on `gpu`, as the A100 campaign's design names them, with the box's own cells
/// beside the core's and the calibration's; the card's are on `host` when one is given.
pub fn cells(gpu: &Gpu, host: Option<&Host>) -> Vec<Cell> {
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
    let name = |s: &str| format!("{} ({})", gpu.title, setting(s, host));
    let measured = format!("{} (measured)", RTX_5060_TI.title);
    let gpu4 = Costs::on(gpu, &QWEN3_4B_FP8, central, host);
    let gpu14 = Costs::on(gpu, &QWEN3_14B_FP8, central, host);
    let own4 = gpu.model_pool_tokens(&QWEN3_4B_FP8);
    let own14 = gpu.model_pool_tokens(&QWEN3_14B_FP8);
    let mut out = vec![
        cell(
            CORE,
            true,
            measured.clone(),
            QWEN3_4B_FP8,
            Costs::frozen_box(),
            24,
            pool,
        ),
        cell(
            CORE,
            true,
            name("the E1 map's"),
            QWEN3_4B_FP8,
            Costs::on(gpu, &QWEN3_4B_FP8, Scaling::MAP, host),
            24,
            pool,
        ),
        cell(
            CORE,
            true,
            name("central"),
            QWEN3_4B_FP8,
            gpu4.clone(),
            24,
            pool,
        ),
        cell(
            CALIBRATION,
            false,
            measured,
            QWEN3_4B_FP8,
            Costs::frozen_box(),
            16,
            pool,
        ),
        cell(
            CALIBRATION,
            false,
            name("central"),
            QWEN3_4B_FP8,
            gpu4.clone(),
            16,
            pool,
        ),
    ];
    out.push(cell(
        MORE_AGENTS,
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

/// The cells with `costs`, the card's fitted costs: the calibration point, to read the fitted model
/// against the measured point, and the discriminating scenario at the box's pool at each of
/// `sessions`, named A' above the core's 24. Each is otherwise its scenario's central cell.
pub fn fitted_cells(gpu: &Gpu, costs: &Costs, sessions: &[usize]) -> Vec<Cell> {
    let cell = |scenario, long, sessions| Cell {
        scenario,
        long,
        label: format!("{} (fitted)", gpu.title),
        model: QWEN3_4B_FP8,
        costs: costs.clone(),
        sessions,
        pool_tokens: lh_core::FROZEN_BOX_POOL_TOKENS,
        cpu_bytes: lh_core::FROZEN_BOX_CPU_BYTES,
    };
    let mut out = vec![cell(CALIBRATION, false, 16)];
    out.extend(
        sessions
            .iter()
            .map(|&s| cell(if s > 24 { MORE_AGENTS } else { CORE }, true, s)),
    );
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

/// The costs table: the box's, and `gpu`'s in each setting for `model`, then on `host` when one is
/// given.
pub fn costs_table(gpu: &Gpu, model: &Model, host: Option<&Host>) -> String {
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
    for on in std::iter::once(None).chain(host.map(Some)) {
        row(
            gpu.title,
            &setting("the E1 map's", on),
            &Costs::on(gpu, model, Scaling::MAP, on),
        );
        for v in Variant::ALL {
            let costs = Costs::on(gpu, model, v.scaling(gpu), on);
            row(gpu.title, &setting(v.name(), on), &costs);
        }
    }
    text
}

/// E0's cells: the box's measured p50s and `gpu`'s forecast, central with the low–high range, the
/// card on `host` when one is given.
pub fn e0_table(gpu: &Gpu, host: Option<&Host>) -> String {
    let mut text = String::from(
        "| Prompt tokens | GPU hit: box | forecast | CPU restore: box | forecast | Recompute: box \
         | forecast |\n|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    let box_costs = Costs::frozen_box();
    let costs = Variant::ALL.map(|v| Costs::on(gpu, &QWEN3_4B_FP8, v.scaling(gpu), host));
    for (i, &tokens) in E0_LENGTHS.iter().enumerate() {
        let base = e0_model(&box_costs, None, tokens);
        let models = costs.each_ref().map(|c| e0_model(c, host, tokens));
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
    format!(
        "Means over seeds {}; `lru-cpu`, the stock engine's policy; a realtime tier of two calls. \
         Running / working set: the time running agent requests / the agents' working set leave \
         those calls short of KV. Busy: the card's step time over the replay's. Stalled: the \
         requests that waited for free blocks.\n\n{}",
        seeds.join(", "),
        table(cells, placed),
    )
}

/// The placements table's head and a row per cell.
fn table(cells: &[Cell], placed: &[Placed]) -> String {
    let mut text = String::from(
        "| Scenario | Card | Sessions | Pool | CPU tier | Pressure | π | ρ | Running / working set \
         | Busy | Stalled | Computed | Resume TTFT p50 / p90 |\n\
         |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
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

/// The cells with the card's fitted costs ([`fitted_cells`]), where they were placed, and the names
/// of the files the costs came from.
#[derive(Clone, Copy, Debug)]
pub struct Fitted<'a> {
    /// E0's summary (`--costs`) and the step-time model (`--step-costs`).
    pub summary: &'a str,
    pub steps: &'a str,
    pub cells: &'a [Cell],
    pub placed: &'a [Placed],
}

/// Of `(sessions, running share)` pairs, the one whose share is closest to `target`, the lower
/// session count on a tie.
pub fn closest_sessions(target: f64, shares: &[(usize, f64)]) -> Option<(usize, f64)> {
    shares.iter().copied().min_by(|a, b| {
        (a.1 - target)
            .abs()
            .total_cmp(&(b.1 - target).abs())
            .then(a.0.cmp(&b.0))
    })
}

/// The fitted cells' section: where their costs came from, their table, and the session count at
/// which the core's running share comes closest to the box's (`cells`' measured core cell).
pub fn fitted_section(fitted: &Fitted<'_>, cells: &[Cell], placed: &[Placed]) -> String {
    let mut text = format!(
        "\n## With the card's fitted costs\n\nThe calibration point and the core at the box's pool \
         with the card's own costs in place of the modeled ones: the step-time model in `{}` and \
         E0's tier fits in `{}`.\n\n",
        fitted.steps, fitted.summary,
    );
    text.push_str(&table(fitted.cells, fitted.placed));
    let on_box = cells
        .iter()
        .zip(placed)
        .find(|(c, _)| c.scenario == CORE && c.label.contains("measured"));
    let core: Vec<(usize, f64)> = fitted
        .cells
        .iter()
        .zip(fitted.placed)
        .filter(|(c, _)| c.long)
        .map(|(c, p)| (c.sessions, p.running_share))
        .collect();
    if let Some((_, b)) = on_box
        && let Some((sessions, share)) = closest_sessions(b.running_share, &core)
    {
        writeln!(
            text,
            "\nClosest to the box's running share in the core ({}): {sessions} sessions ({}).",
            pct(b.running_share),
            pct(share),
        )
        .expect("writing to a String");
    }
    text
}

/// The host the card is placed on (`--host`), and the name of the file that describes it.
#[derive(Clone, Copy, Debug)]
pub struct OnHost<'a> {
    pub host: &'a Host,
    /// bench_host.py's `host.json`.
    pub file: &'a str,
}

/// The whole report for `gpu`, on `host` when one is given, with the fitted cells' section after
/// the scenario table when `fitted` gives them.
pub fn report(
    gpu: &Gpu,
    host: Option<&OnHost<'_>>,
    cells: &[Cell],
    placed: &[Placed],
    seeds: &[u64],
    fitted: Option<&Fitted<'_>>,
) -> String {
    // The make target that writes it: the A100 40 GB's is the default.
    let mut target = if gpu.name == hardware::A100_40GB.name {
        "make forecast-a100".to_string()
    } else {
        format!("make forecast-a100 GPU={}", gpu.name)
    };
    let path = format!(
        "the path vLLM 0.32.0 takes for the box's checkpoint on it ({}; \
         `crates/lh-sim/src/hardware.rs`)",
        gpu.linear.label()
    );
    // What the costs come from, and on a host what it sets.
    let (sources, host_terms) = match host {
        None => (
            format!("the card's published peaks and {path}"),
            String::new(),
        ),
        Some(h) => {
            write!(target, " HOST_JSON={}", h.file).expect("writing to a String");
            let cpu = match &h.host.cpu {
                Some(name) => format!(" ({name})"),
                None => String::new(),
            };
            (
                format!(
                    "the card's published peaks, {path} and the host it is placed on, as \
                     `benchmarks/a100/bench_host.py` measured it: its CPU{cpu} runs the Python \
                     workload in {:.1} ms against the box's {:.1}, and its link copies pinned host \
                     memory to the card at {:.2} GB/s in one copy against the box's {:.2}",
                    h.host.python_ms,
                    hardware::BOX_PYTHON_MS,
                    h.host.h2d_gb_per_s,
                    hardware::BOX_H2D_GB_PER_S,
                ),
                format!(
                    " On this host the API server's time is the box's over {:.2}, and a restore's \
                     transfer runs at {:.0}% of the host's one-copy rate, as the box's E0 restore \
                     does of the box's; the restore's fixed part, the mixed-step extra and the \
                     restore stall stay the box's.",
                    h.host.cpu_ratio(),
                    100.0 * hardware::BOX_RESTORE_SHARE,
                ),
            )
        }
    };
    let on_this_host = host.map_or("", |_| " on this host");
    let host = host.map(|h| h.host);
    let mut text = String::new();
    writeln!(
        text,
        "# Forecast: the {} against the frozen box\n\n\
         Written by `lh-sim forecast` (`{target}`) from the box's measured costs, {sources}. Every \
         number for the {} is modeled; the box's are measured where they say so.{host_terms} Three \
         settings bound it: **low**, the peaks' ratios alone with Marlin as fast as a BF16 GEMM; \
         **central**, with the box's 1.7 ms of step overhead kept, Marlin {:.2}× a BF16 GEMM (vLLM \
         PR #61224's L4) and FlashAttention 2 at its best published efficiency on the card; \
         **high**, with FlashAttention 2 at its worst published. The E1 map's model, on which \
         `lh-sim hardware` places every card, scales the fixed step with bandwidth and keeps the \
         box's attention efficiency.\n",
        gpu.title,
        gpu.title,
        hardware::MARLIN_FACTOR,
    )
    .expect("writing to a String");
    writeln!(text, "## Step and tier costs, Qwen3-4B-Instruct-2507-FP8\n")
        .expect("writing to a String");
    text.push_str(&costs_table(gpu, &QWEN3_4B_FP8, host));
    if gpu.model_pool_tokens(&QWEN3_14B_FP8) > 0 {
        writeln!(text, "\n## Step and tier costs, Qwen3-14B-FP8\n").expect("writing to a String");
        text.push_str(&costs_table(gpu, &QWEN3_14B_FP8, host));
    }
    let ratio = host.map_or("", |_| ", the card on this host and the box on its own");
    writeln!(
        text,
        "\n## E0's cells\n\nTime to first token of a request alone, in ms: the box's measured p50 \
         (2026-10-22), and the forecast as that times the modeled ratio of the two cards{ratio}, \
         central (low–high).\n"
    )
    .expect("writing to a String");
    text.push_str(&e0_table(gpu, host));
    writeln!(text, "\n## The engine probes\n").expect("writing to a String");
    text.push_str(&probes_section(gpu));
    writeln!(
        text,
        "\n## E2's scenarios on the regime map\n\nEach cell replays E2's agents in `lh-sim`'s model \
         of vLLM 0.32.0's two-tier KV cache and step loop with the card's central \
         costs{on_this_host}; the discriminating scenario replays `weka-x32-long`, the calibration \
         point the whole 32x corpus.\n"
    )
    .expect("writing to a String");
    text.push_str(&placements_table(cells, placed, seeds));
    if let Some(fitted) = fitted {
        text.push_str(&fitted_section(fitted, cells, placed));
    }
    writeln!(text, "\n## What the forecast says the run will show\n").expect("writing to a String");
    text.push_str(&reading(gpu, cells, placed));
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{A100_40GB, A100_80GB, L4};

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
        let table = e0_table(&RTX_5060_TI, None);
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
        let cells = cells(&A100_40GB, None);
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
            !super::cells(&RTX_5060_TI, None)
                .iter()
                .any(|c| c.model == QWEN3_14B_FP8)
        );
    }

    #[test]
    fn the_tables_carry_the_settings() {
        let t = costs_table(&A100_40GB, &QWEN3_4B_FP8, None);
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
        let t14 = costs_table(&A100_40GB, &QWEN3_14B_FP8, None);
        assert!(!t14.contains("measured"), "{t14}");
        let probes = probes_section(&A100_40GB);
        assert!(probes.contains("1,536 tokens (central"), "{probes}");
        assert!(probes.contains("on Gray 4,923–5,077 ms"), "{probes}");
    }

    #[test]
    fn the_a100_80gbs_forecast_is_the_40gbs_with_its_bandwidth_and_pools() {
        // The step costs: the fixed step and the KV reads by the bandwidth, the rest the 40 GB card's.
        let t = costs_table(&A100_80GB, &QWEN3_4B_FP8, None);
        for row in [
            "| A100 SXM4 80 GB | the E1 map's | 2.54 ms | 41.33 µs | 2.15 ns | 94.3 ns |",
            "| A100 SXM4 80 GB | low | 2.54 ms | 29.19 µs | 2.15 ns | 94.3 ns |",
            "| A100 SXM4 80 GB | central | 3.88 ms | 41.33 µs | 2.59 ns | 94.3 ns |",
            "| A100 SXM4 80 GB | high | 3.88 ms | 41.33 µs | 3.78 ns | 94.3 ns |",
        ] {
            assert!(t.contains(row), "{row}\n{t}");
        }
        assert!(t.contains("| 10.3 ms + 5.91 µs/token | 3,123 ms |"), "{t}");
        // E0's cells at 31,744 tokens: a GPU hit, a restore and a recompute, each a little faster.
        let e0 = e0_table(&A100_80GB, None);
        assert!(
            e0.contains(
                "| 31,744 | 156 | 131 (130–132) | 903 | 324 (322–324) | 10,239 | 3,169 (2,459–3,788) |"
            ),
            "{e0}"
        );
        // The KV case's wait is two 18k decodes at a time: 7.5 ms on this card against the 40 GB
        // card's 9.2, so stock waits 1.3-1.4 s; the caps are the 40 GB card's.
        let central = Costs::of(&A100_80GB, &QWEN3_4B_FP8, Scaling::central(&A100_80GB));
        assert!((two_decodes_ms(&central.steps) - 7.45).abs() < 0.01);
        let probes = probes_section(&A100_80GB);
        for text in [
            "On the A100 SXM4 80 GB: 1,328–1,369 ms (low–high 1,082–1,377).",
            "an idle pool: 126–145 ms (109–146).",
            "On the A100 SXM4 80 GB: 32.8 ms (25.7–35.4) and 5.6 ms (3.7–5.6)",
            "1,536 tokens (central, 127 ms)",
        ] {
            assert!(probes.contains(text), "{text}\n{probes}");
        }
        // The core's cells are at the box's pool, as on the 40 GB card; C' and D at the card's own.
        let cells = cells(&A100_80GB, None);
        let names: Vec<_> = cells
            .iter()
            .map(|c| (c.scenario, c.sessions, c.pool_tokens))
            .collect();
        assert_eq!(names[2], ("A: the core, the box's pool", 24, 36_400));
        assert_eq!(cells[2].label, "A100 SXM4 80 GB (central)");
        assert!(names.contains(&("C': Qwen3-14B-FP8 at its own pool", 40, 379_488)));
        assert!(names.contains(&("D: Qwen3-4B at the card's own pool", 24, 502_432)));
        // The report names the make target that writes it.
        let text = report(&A100_80GB, None, &cells, &[], &[11], None);
        assert!(
            text.starts_with(
                "# Forecast: the A100 SXM4 80 GB against the frozen box\n\nWritten by `lh-sim \
                 forecast` (`make forecast-a100 GPU=a100-80gb`)"
            ),
            "{text}"
        );
        assert!(
            report(&A100_40GB, None, &[], &[], &[11], None).contains("(`make forecast-a100`) from")
        );
    }

    #[test]
    fn the_fitted_cells_are_the_central_cells_at_the_boxs_pool_with_the_fitted_costs() {
        let costs = Costs::of(&A100_80GB, &QWEN3_4B_FP8, Scaling::LOW);
        let fitted = fitted_cells(&A100_80GB, &costs, &[24, 28, 32, 36, 40]);
        let names: Vec<_> = fitted
            .iter()
            .map(|c| (c.scenario, c.long, c.sessions))
            .collect();
        assert_eq!(
            names,
            [
                ("the calibration point, the whole corpus", false, 16),
                ("A: the core, the box's pool", true, 24),
                ("A': the box's pool, more agents", true, 28),
                ("A': the box's pool, more agents", true, 32),
                ("A': the box's pool, more agents", true, 36),
                ("A': the box's pool, more agents", true, 40),
            ]
        );
        // Each is its scenario's central cell with the fitted costs in its place: the box's pool
        // and 20 GiB CPU tier, the box's checkpoint.
        let modeled = cells(&A100_80GB, None);
        for f in &fitted {
            assert_eq!(f.label, "A100 SXM4 80 GB (fitted)");
            assert_eq!(f.costs, costs);
            let c = modeled
                .iter()
                .find(|c| c.scenario == f.scenario && c.label.contains("central"))
                .unwrap();
            assert_eq!(
                (f.long, f.model, f.pool_tokens, f.cpu_bytes),
                (c.long, c.model, c.pool_tokens, c.cpu_bytes)
            );
        }
        assert_eq!(
            (fitted[1].pool_tokens, fitted[1].cpu_bytes),
            (36_400, 20 << 30)
        );
        // The calibration point alone when no count of the core is asked.
        assert_eq!(fitted_cells(&A100_80GB, &costs, &[]).len(), 1);
    }

    #[test]
    fn the_closest_session_count_takes_the_lower_on_a_tie() {
        let shares = [(24, 0.10), (28, 0.15), (32, 0.18), (36, 0.25)];
        assert_eq!(closest_sessions(0.17, &shares), Some((32, 0.18)));
        // A quarter and three quarters are as far from a half: the lower count, wherever it is.
        assert_eq!(
            closest_sessions(0.5, &[(32, 0.75), (28, 0.25)]),
            Some((28, 0.25))
        );
        assert_eq!(closest_sessions(0.5, &[]), None);
    }

    #[test]
    fn the_report_has_the_fitted_section_only_with_the_fitted_cells() {
        let at = |running_share: f64| Placed {
            pressure_mean: 9.0,
            pressure_peak: 10.0,
            pi: 98.0,
            rho: 0.11,
            running_share,
            working_set_share: 1.0,
            busy: 0.71,
            stalled: 0.5,
            computed: 0.4,
            resume_p50_ms: 1_168.0,
            resume_p90_ms: 9_717.0,
        };
        let modeled = cells(&A100_80GB, None);
        // The box's core cell comes first, at 17%.
        let placed: Vec<_> = (0..modeled.len())
            .map(|i| at(if i == 0 { 0.17 } else { 0.1 }))
            .collect();
        let without = report(&A100_80GB, None, &modeled, &placed, &[11], None);
        assert!(
            !without.contains("## With the card's fitted costs"),
            "{without}"
        );
        // The calibration point at the box's share, which the closest count leaves out, then the
        // core from 24 sessions to 40.
        let cells = fitted_cells(&A100_80GB, &Costs::frozen_box(), &[24, 28, 32, 36, 40]);
        let shares = [0.17, 0.10, 0.15, 0.18, 0.21, 0.30];
        let fitted_placed: Vec<_> = shares.iter().map(|&s| at(s)).collect();
        let fitted = Fitted {
            summary: "summary.json",
            steps: "steps.json",
            cells: &cells,
            placed: &fitted_placed,
        };
        let with = report(&A100_80GB, None, &modeled, &placed, &[11], Some(&fitted));
        // The section comes between the scenario table and the reading, and the rest is unchanged.
        let start = with.find("\n## With the card's fitted costs\n").unwrap();
        let end = with
            .find("\n## What the forecast says the run will show\n")
            .unwrap();
        assert_eq!(format!("{}{}", &with[..start], &with[end..]), without);
        let section = &with[start..end];
        assert!(
            section.contains(
                "with the card's own costs in place of the modeled ones: the step-time model in \
                 `steps.json` and E0's tier fits in `summary.json`.\n\n| Scenario | Card |"
            ),
            "{section}"
        );
        assert!(
            section.contains(
                "| A': the box's pool, more agents | A100 SXM4 80 GB (fitted) | 32 | 36,400 tokens \
                 | 20 GiB | 9.0 / 10.0 | 98.0 | 0.11 | 18% / 100% | 71% | 50% | 40% \
                 | 1,168 / 9,717 ms |\n"
            ),
            "{section}"
        );
        assert_eq!(section.matches(" (fitted) |").count(), 6);
        assert!(
            section.ends_with(
                "|\n\nClosest to the box's running share in the core (17%): 32 sessions (18%).\n"
            ),
            "{section}"
        );
        // Without the box's core cell there is no share to come closest to.
        let alone = fitted_section(&fitted, &modeled[1..], &placed[1..]);
        assert!(!alone.contains("Closest"), "{alone}");
    }

    /// The box's host as bench_host.py measured it on Phastform.
    fn box_host() -> Host {
        Host::from_bench_host(include_str!("../testdata/box-host.json")).unwrap()
    }

    /// A stand-in for the A100 run's host: twice the box's speed at Python, 12.1 GB/s in one copy.
    fn a100_host() -> Host {
        Host {
            cpu: Some("Intel(R) Xeon(R) CPU @ 2.20GHz".into()),
            python_ms: 190.0,
            h2d_gb_per_s: 12.1,
        }
    }

    #[test]
    fn the_boxs_own_host_leaves_the_boxs_forecast_as_it_is() {
        let b = box_host();
        for s in [
            Scaling::MAP,
            Scaling::LOW,
            Scaling::central(&RTX_5060_TI),
            Scaling::high(&RTX_5060_TI),
        ] {
            assert_eq!(
                Costs::on(&RTX_5060_TI, &QWEN3_4B_FP8, s, Some(&b)),
                Costs::frozen_box()
            );
        }
        assert_eq!(
            e0_table(&RTX_5060_TI, Some(&b)),
            e0_table(&RTX_5060_TI, None)
        );
        // Its cells are the ones without a host, which say where they are.
        let (on, without) = (cells(&RTX_5060_TI, Some(&b)), cells(&RTX_5060_TI, None));
        assert_eq!(on.len(), without.len());
        for (c, w) in on.iter().zip(&without) {
            assert_eq!(c.costs, w.costs);
            assert_eq!(c.label.replace(", on this host", ""), w.label);
        }
        assert_eq!(on[2].label, "RTX 5060 Ti 16 GB (central, on this host)");
    }

    #[test]
    fn on_a_host_the_cards_cells_take_its_costs_and_the_report_says_so() {
        let h = a100_host();
        let on = Some(&h);
        // Each costs table keeps its rows and adds the card's on the host: its link's 13.21 µs a
        // token for the 4B's KV (14.68 for the 14B's), and a cold prefill shorter by E0's host time
        // saved.
        let (with, without) = (
            costs_table(&A100_80GB, &QWEN3_4B_FP8, on),
            costs_table(&A100_80GB, &QWEN3_4B_FP8, None),
        );
        assert!(with.starts_with(&without), "{with}");
        assert_eq!(with.matches(", on this host |").count(), 4);
        for row in [
            "| A100 SXM4 80 GB | the E1 map's, on this host | 2.54 ms | 41.33 µs | 2.15 ns | 94.3 ns \
             | 10.3 ms + 13.21 µs/token | 2,754 ms |",
            "| A100 SXM4 80 GB | central, on this host | 3.88 ms | 41.33 µs | 2.59 ns | 94.3 ns \
             | 10.3 ms + 13.21 µs/token | 3,065 ms |",
        ] {
            assert!(with.contains(row), "{row}\n{with}");
        }
        let t14 = costs_table(&A100_80GB, &QWEN3_14B_FP8, on);
        assert!(
            t14.contains(
                "| A100 SXM4 80 GB | central, on this host | 8.96 ms | 150.21 µs | 3.60 ns \
                 | 104.8 ns | 10.3 ms + 14.68 µs/token | 7,379 ms |"
            ),
            "{t14}"
        );
        // The box's own cells are as they were; the card's are on the host, and say so.
        let (on_host, without) = (cells(&A100_80GB, on), cells(&A100_80GB, None));
        assert_eq!(on_host.len(), without.len());
        for (c, w) in on_host.iter().zip(&without) {
            if c.label.contains("measured") {
                assert_eq!((&c.label, &c.costs), (&w.label, &w.costs));
                continue;
            }
            assert_eq!(c.label, w.label.replace(')', ", on this host)"));
            let scaling = if c.label.contains("the E1 map's") {
                Scaling::MAP
            } else {
                Scaling::central(&A100_80GB)
            };
            assert_eq!(c.costs, Costs::on(&A100_80GB, &c.model, scaling, on));
            assert_ne!(c.costs, w.costs);
        }
        // There the API server's time is the box's over 2.01: 14.76 ms + 3.985 µs a prompt token,
        // and 3.24 ms for each request decoding.
        let s = &on_host[2].costs.steps;
        assert!((s.host.ms - 14.76).abs() < 0.005 && (s.host.us_per_token - 3.985).abs() < 0.001);
        assert!((s.host_per_decode_ms - 3.24).abs() < 0.005);
        // E0's cells: the GPU hits about half the forecast's without a host, the restores slower.
        let e0 = e0_table(&A100_80GB, on);
        for row in [
            "| 1,024 | 45.0 | 20.3 (18.6–20.4) | 64.0 | 36.1 (34.8–36.1) | 135 | 82.0 (62.4–82.7) |",
            "| 31,744 | 156 | 70.0 (68.3–70.6) | 903 | 492 (491–493) | 10,239 | 3,106 (2,397–3,725) |",
        ] {
            assert!(e0.contains(row), "{row}\n{e0}");
        }
        // The report names the host where it says what the costs come from, and says where E0's
        // cells and the scenarios' are.
        let file = OnHost {
            host: &h,
            file: "host.json",
        };
        let text = report(&A100_80GB, Some(&file), &on_host, &[], &[11], None);
        for part in [
            "(`make forecast-a100 GPU=a100-80gb HOST_JSON=host.json`) from the box's measured costs, the \
             card's published peaks, the path vLLM 0.32.0 takes for the box's checkpoint on it \
             (Marlin W8A16; `crates/lh-sim/src/hardware.rs`) and the host it is placed on, as \
             `benchmarks/a100/bench_host.py` measured it: its CPU (Intel(R) Xeon(R) CPU @ 2.20GHz) \
             runs the Python workload in 190.0 ms against the box's 381.1, and its link copies \
             pinned host memory to the card at 12.10 GB/s in one copy against the box's 6.76. Every \
             number for the A100 SXM4 80 GB is modeled; the box's are measured where they say so. \
             On this host the API server's time is the box's over 2.01, and a restore's transfer \
             runs at 92% of the host's one-copy rate, as the box's E0 restore does of the box's; \
             the restore's fixed part, the mixed-step extra and the restore stall stay the box's. \
             Three settings bound it:",
            "the modeled ratio of the two cards, the card on this host and the box on its own, \
             central (low–high).",
            "step loop with the card's central costs on this host; the discriminating scenario",
        ] {
            assert!(text.contains(part), "{part}\n{text}");
        }
        // A file without the CPU's name gives the sentence without it.
        let unnamed = Host { cpu: None, ..h };
        let file = OnHost {
            host: &unnamed,
            file: "host.json",
        };
        let text = report(&A100_80GB, Some(&file), &[], &[], &[11], None);
        assert!(
            text.contains("measured it: its CPU runs the Python workload in 190.0 ms"),
            "{text}"
        );
    }
}
