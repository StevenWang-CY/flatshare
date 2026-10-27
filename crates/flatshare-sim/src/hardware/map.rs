//! The cards on the E1 regime map: `flatshare-sim hardware`. [`sweep`] runs the E1 arms in each GPU's cells, at the
//! GPU's pool and with its modeled costs ([`super`]), and records where it sits on the map (KV pressure, π, ρ), how
//! long the agents leave a realtime tier short of KV, and the step a realtime turn waits behind; [`csv`],
//! [`profiles`] and [`markdown`] are the command's tables. Each GPU replays the corpus at one token scale, the
//! same for every card or its own ([`fitted_token_scale`]).

use std::fmt::Write as _;
use std::path::Path;

use flatshare_core::{FROZEN_BOX_POOL_TOKENS, KvGeometry, StepCost, StepLoad};

use super::{Gpu, LinearPath, MARLIN_FACTOR};
use crate::aiperf::PlanOptions;
use crate::corpus::Corpus;
use crate::metrics::Summary;
use crate::regime::{self, signed, thousands};
use crate::replay::{Pauses, ReplayConfig};
use crate::run::Arm;
use crate::scheduler::StepBudget;

/// Qwen3-4B-Instruct-2507's native context
/// (<https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8>): vLLM serves no longer request of it.
pub const MODEL_CONTEXT: u64 = 262_144;

/// The regime map's largest session count, where the box is at pressure 7 to 7.5.
const BOX_TOP_SESSIONS: f64 = 40.0;

/// The step of one 512-token background prefill chunk after `context` cached tokens, alone: what a
/// realtime request that arrives during it waits behind, in ms.
pub fn chunk_ms(steps: &StepCost, context: u64) -> f64 {
    let mut load = StepLoad::default();
    load.add_prefill(StepBudget::FROZEN_BOX.max_batched_tokens, context);
    steps.step_ms(&load)
}

/// The largest prompt plus output of a request of `corpus`, in tokens at token scale 1: at token
/// scale `k`, a request needs at most `k` times this, less one, of KV.
pub fn longest_request(corpus: &mut Corpus) -> Result<u64, String> {
    let mut longest = 0;
    for i in 0..corpus.len() {
        let plan = corpus.plan(i)?;
        for t in plan.convs.iter().flat_map(|c| &c.turns) {
            let tokens = u64::from(t.hash_blocks) * u64::from(plan.block_size);
            longest = longest.max(tokens + u64::from(t.max_tokens));
        }
    }
    Ok(longest)
}

/// The token scale closest to the traces' original token counts that `gpu` can serve: the largest
/// power of two up to `max_scale` at which a request of `longest` tokens at token scale 1
/// ([`longest_request`]) fits both its pool and the model's context, as the 32x replay fits the box.
/// `None` when it fits at no scale.
pub fn fitted_token_scale(
    gpu: &Gpu,
    geometry: &KvGeometry,
    longest: u64,
    max_scale: u32,
) -> Option<u32> {
    // vLLM's null block holds no KV.
    let usable = (gpu.pool_tokens(geometry) / geometry.block_tokens).saturating_sub(1);
    let fits = |k: u32| {
        let tokens = u64::from(k) * longest;
        tokens <= MODEL_CONTEXT
            && tokens.saturating_sub(1).div_ceil(geometry.block_tokens) <= usable
    };
    std::iter::successors(Some(1u32), |k| k.checked_mul(2))
        .take_while(|&k| k <= max_scale && fits(k))
        .last()
}

/// The counts of `sessions` that `gpu` runs at `token_scale`: those up to 40·r, where r is its pool
/// over the box's in the replay's tokens. A pool r times larger meets the box's pressures at about r
/// times the sessions, and the box itself runs the regime map's counts.
pub fn sessions_for(
    gpu: &Gpu,
    geometry: &KvGeometry,
    token_scale: u32,
    sessions: &[usize],
) -> Vec<usize> {
    let box_pool = (FROZEN_BOX_POOL_TOKENS * u64::from(token_scale)) as f64;
    let r = gpu.pool_tokens(geometry) as f64 / box_pool;
    sessions
        .iter()
        .copied()
        .filter(|&s| s as f64 <= BOX_TOP_SESSIONS * r.max(1.0))
        .collect()
}

/// One cell of [`sweep`]: a GPU, the token scale and seed of its replay, and the regime cell (whose
/// capacity is the GPU's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub gpu: Gpu,
    pub token_scale: u32,
    pub seed: u64,
    pub regime: regime::Cell,
}

/// The cells of `gpus`, each at its token scale, over the counts of `sessions` it runs and every
/// seed.
pub fn cells(
    gpus: &[(Gpu, u32)],
    sessions: &[usize],
    seeds: &[u64],
    pauses: Pauses,
    cpu_ratio: f64,
) -> Vec<Cell> {
    let geometry = KvGeometry::QWEN3_4B_BF16;
    let mut out = Vec::new();
    for &(gpu, token_scale) in gpus {
        for n in sessions_for(&gpu, &geometry, token_scale, sessions) {
            for &seed in seeds {
                out.push(Cell {
                    gpu,
                    token_scale,
                    seed,
                    regime: regime::Cell {
                        pauses,
                        sessions: n,
                        cpu_ratio,
                        capacity: gpu.capacity(&geometry, cpu_ratio),
                    },
                });
            }
        }
    }
    out
}

/// A cell of [`sweep`] and what the E1 arms did in it, at the cell's seed.
#[derive(Clone, Debug)]
pub struct Placement {
    pub gpu: Gpu,
    pub token_scale: u32,
    pub result: regime::CellResult,
}

/// Runs the E1 arms in every cell on `jobs` threads, each GPU with its own costs. `base` is the rest
/// of the replay; its concurrency, seed, token scale and pauses give way to each cell's.
pub fn sweep(
    traces: &Path,
    plan: PlanOptions,
    base: &ReplayConfig,
    cells: &[Cell],
    jobs: usize,
) -> Result<Vec<Placement>, String> {
    regime::par_map(traces, plan, cells, jobs, |corpus, cell| {
        let replay = ReplayConfig {
            concurrency: cell.regime.sessions,
            seed: cell.seed,
            token_scale: cell.token_scale,
            pauses: cell.regime.pauses,
            ..base.clone()
        };
        let result = regime::run_cell(corpus, &cell.gpu.run_config(replay), cell.regime)
            .map_err(|e| format!("{}: {e}", cell.gpu.name))?;
        Ok(Placement {
            gpu: cell.gpu,
            token_scale: cell.token_scale,
            result,
        })
    })
}

fn opt(x: Option<f64>) -> String {
    x.map_or(String::new(), |v| format!("{v:.4}"))
}

/// One row per cell and arm: the regime sweep's columns between the GPU's and the realtime tier's.
pub fn csv(results: &[Placement]) -> String {
    let mut text = format!(
        "gpu,costs,token_scale,pool_tokens,{},kv_pressure_peak,median_prompt_tokens,\
         realtime_short_share,realtime_blocked_share,chunk_ms\n",
        regime::CSV_COLUMNS
    );
    let block_tokens = KvGeometry::QWEN3_4B_BF16.block_tokens;
    for p in results {
        let steps = p.gpu.step_cost();
        for (s, &arm) in p.result.arms.iter().zip(&Arm::E1) {
            write!(
                text,
                "{},{},{},{},",
                p.gpu.name,
                p.gpu.label(),
                p.token_scale,
                p.result.cell.capacity.gpu_blocks * block_tokens,
            )
            .expect("writing to a String");
            regime::csv_row(&mut text, &p.result, s, arm);
            let median = s.regime.median_prompt_tokens;
            writeln!(
                text,
                ",{:.3},{median:.0},{},{},{:.1}",
                s.regime.kv_pressure_peak,
                opt(s.replay.realtime_short_share),
                opt(s.replay.realtime_blocked_share),
                chunk_ms(&steps, median.round() as u64),
            )
            .expect("writing to a String");
        }
    }
    text
}

/// `x` to a tenth, with thousands separators.
fn tenths(x: f64) -> String {
    let t = (x * 10.0).round() as u64;
    format!("{}.{}", thousands(t / 10), t % 10)
}

/// `mib` in GiB: whole, or to a tenth.
fn gib(mib: u64) -> String {
    if mib.is_multiple_of(1024) {
        thousands(mib / 1024)
    } else {
        tenths(mib as f64 / 1024.0)
    }
}

/// A dense rate in TFLOPS to a tenth, or "none" for a card without it.
fn rate(tflops: f64) -> String {
    if tflops > 0.0 {
        tenths(tflops)
    } else {
        "none".into()
    }
}

/// The GPUs: their published peaks and the costs modeled from them.
pub fn profiles(gpus: &[Gpu]) -> String {
    let geometry = KvGeometry::QWEN3_4B_BF16;
    let mut text = String::from(
        "| GPU | Costs | Memory | Bandwidth | Dense FP8 / BF16 | Linear layers | Host link, each way \
         | Pool | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill |\n\
         |---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|\n",
    );
    for g in gpus {
        let (s, t) = (g.step_cost(), g.tier_cost());
        let c = &t.recompute;
        let linear = match g.linear {
            LinearPath::Fp8 => LinearPath::Fp8.label().to_string(),
            LinearPath::Marlin => format!("{}, {MARLIN_FACTOR:.2}× a BF16 GEMM", g.linear.label()),
        };
        writeln!(
            text,
            "| {} | {} | {} GiB | {} GB/s | {} / {} TFLOPS | {linear} | PCIe {}.0 x{}, {:.1} GB/s \
             | {} tokens | {:.2} ms | {:.2} µs | {:.2} ns | {:.1} ns | {:.1} ms + {:.2} µs/token \
             | {:.1} ms + {:.2} µs/token + {:.3} ns/token² |",
            g.title,
            g.label(),
            gib(g.memory_mib),
            thousands(g.bandwidth_gb_s as u64),
            rate(g.fp8_tflops),
            tenths(g.bf16_tflops),
            g.link.generation,
            g.link.lanes,
            g.link.gb_per_s(),
            thousands(g.pool_tokens(&geometry)),
            s.step_ms,
            s.us_per_token,
            s.ns_per_pair,
            s.ns_per_context,
            t.cpu_restore.ms,
            t.cpu_restore.us_per_token,
            c.ms,
            c.us_per_token,
            c.ns_per_token_squared,
        )
        .expect("writing to a String");
    }
    text
}

/// The placements: per GPU and token scale, one row per session count, each the mean over its seeds.
/// `hash_block` is the corpus's tokens per hash block, for the scaling against the original traces'
/// 64.
pub fn markdown(results: &[Placement], hash_block: u32, realtime_tokens: u64) -> String {
    fn lru(p: &Placement) -> &Summary {
        p.result.arm(Arm::LruCpu)
    }
    let mut groups: Vec<Vec<&Placement>> = Vec::new();
    for p in results {
        let key = |q: &Placement| (q.gpu, q.token_scale, q.result.cell);
        match groups.iter_mut().find(|g| key(g[0]) == key(p)) {
            Some(g) => g.push(p),
            None => groups.push(vec![p]),
        }
    }
    let Some(first) = groups.first() else {
        return String::new();
    };
    let seeds: Vec<_> = first.iter().map(|p| p.result.seed.to_string()).collect();
    let seeds = match &seeds[..] {
        [one] => format!("seed {one}"),
        all => format!("means over seeds {}", all.join(", ")),
    };
    let mut text = String::new();
    writeln!(
        text,
        "**Pauses: {}; the CPU tier {}× the pool; {seeds}.** Pressure: mean / peak. Voice KV short: \
         the share of time the live working set / the running requests' KV leave less than {} \
         tokens of the pool free. Chunk: one 512-token background prefill chunk at the median \
         prompt, in ms. Resume TTFT p50 / p90 in ms; gains against `lru-cpu` at p50 / p90.\n\n\
         | GPU | Sessions | Pressure | π | ρ | Voice KV short | Chunk | `lru-cpu` | `clairvoyant` \
         | `oracle-prefetch` | `history-prefetch` | `tokencake` |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        regime::pauses_name(first[0].result.cell.pauses),
        first[0].result.cell.cpu_ratio,
        thousands(realtime_tokens),
    )
    .expect("writing to a String");
    let mut previous = None;
    for g in &groups {
        let p = g[0];
        let n = g.len() as f64;
        let mean = |f: &dyn Fn(&Placement) -> f64| g.iter().map(|q| f(q)).sum::<f64>() / n;
        let share = |f: &dyn Fn(&Placement) -> Option<f64>| {
            g.iter().map(|q| f(q)).sum::<Option<f64>>().map(|x| x / n)
        };
        let short = share(&|q| lru(q).replay.realtime_short_share);
        let blocked = share(&|q| lru(q).replay.realtime_blocked_share);
        let steps = p.gpu.step_cost();
        let ms = |pct| thousands(mean(&|q| q.result.resume(Arm::LruCpu, pct)).round() as u64);
        let gains = |arm: Arm| {
            format!(
                "{} / {}%",
                signed(100.0 * mean(&|q| q.result.gain(arm, 50)), 0),
                signed(100.0 * mean(&|q| q.result.gain(arm, 90)), 0)
            )
        };
        let gpu = if previous == Some((p.gpu.name, p.token_scale)) {
            String::new()
        } else {
            let scale = 64.0 / f64::from(hash_block * p.token_scale);
            format!("{} ({}), {scale}×", p.gpu.title, p.gpu.label())
        };
        previous = Some((p.gpu.name, p.token_scale));
        writeln!(
            text,
            "| {gpu} | {} | {:.1} / {:.1} | {:.1} | {:.2} | {} | {:.0} | {} / {} | {} | {} | {} | {} |",
            p.result.cell.sessions,
            mean(&|q| lru(q).regime.kv_pressure_mean),
            mean(&|q| lru(q).regime.kv_pressure_peak),
            mean(&|q| lru(q).regime.pi),
            mean(&|q| lru(q).regime.rho),
            match (short, blocked) {
                (Some(s), Some(b)) => format!("{:.0} / {:.0}%", 100.0 * s, 100.0 * b),
                _ => "–".into(),
            },
            mean(&|q| chunk_ms(&steps, lru(q).regime.median_prompt_tokens.round() as u64)),
            ms(50),
            ms(90),
            gains(Arm::Clairvoyant),
            gains(Arm::OraclePrefetch),
            gains(Arm::HistoryPrefetch),
            gains(Arm::Tokencake),
        )
        .expect("writing to a String");
    }
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{A100_40GB, GPUS, H100_SXM, L4, RTX_5060_TI};
    use crate::metrics::{Percentiles, Regime};
    use crate::replay::ReplayStats;

    const QWEN: KvGeometry = KvGeometry::QWEN3_4B_BF16;

    #[test]
    fn a_corpus_replays_at_the_least_scaling_its_pool_and_the_context_hold() {
        // The 32x corpus's longest request: 31,144 tokens of prompt and output.
        let longest = 31_144;
        let fitted = |g: &Gpu| fitted_token_scale(g, &QWEN, longest, 32);
        // The box holds it as given; the L4 at 16x (twice the tokens); the A100 at 8x; the H100 at
        // 4x, where the model's context binds before its pool does.
        assert_eq!(
            (
                fitted(&RTX_5060_TI),
                fitted(&L4),
                fitted(&A100_40GB),
                fitted(&H100_SXM)
            ),
            (Some(1), Some(2), Some(4), Some(8))
        );
        assert!(16 * longest > MODEL_CONTEXT && 16 * longest < H100_SXM.pool_tokens(&QWEN));
        // No scale goes past the original token counts, and the original corpus fits no pool.
        assert_eq!(fitted_token_scale(&H100_SXM, &QWEN, longest, 4), Some(4));
        assert_eq!(fitted_token_scale(&H100_SXM, &QWEN, 996_579, 1), None);
        // The box's 2,274 usable blocks hold 36,384 tokens: a request of 36,385 (36,384 of KV) fits.
        assert_eq!(fitted_token_scale(&RTX_5060_TI, &QWEN, 36_385, 32), Some(1));
        assert_eq!(fitted_token_scale(&RTX_5060_TI, &QWEN, 36_386, 32), None);
        // A card with no room beside the box's footprint has no pool at all.
        let small = Gpu {
            name: "test",
            memory_mib: 8 << 10,
            ..RTX_5060_TI
        };
        assert_eq!(small.pool_tokens(&QWEN), 0);
        assert_eq!(fitted_token_scale(&small, &QWEN, 16, 32), None);
    }

    #[test]
    fn a_larger_pool_runs_more_sessions() {
        let all = [5, 8, 12, 16, 24, 32, 40, 64, 96, 128, 192, 256];
        let regime_map = [5, 8, 12, 16, 24, 32, 40];
        assert_eq!(sessions_for(&RTX_5060_TI, &QWEN, 1, &all), regime_map);
        // 2.3, 5.8 and 13.8 times the box's pool at its scale: up to 92, 232 and 552 sessions.
        assert_eq!(sessions_for(&L4, &QWEN, 1, &all).last(), Some(&64));
        assert_eq!(sessions_for(&A100_40GB, &QWEN, 1, &all).last(), Some(&192));
        assert_eq!(sessions_for(&H100_SXM, &QWEN, 1, &all).len(), all.len());
        // At 4x, the H100's pool holds 1.7 times the box's at 32x: up to 69 sessions.
        assert_eq!(sessions_for(&H100_SXM, &QWEN, 8, &all).last(), Some(&64));
        let c = cells(
            &[(RTX_5060_TI, 1), (L4, 2)],
            &all,
            &[11, 12],
            Pauses::Trace,
            4.0,
        );
        assert_eq!(c.len(), 2 * (7 + 7));
        assert_eq!((c[0].regime.sessions, c[0].seed, c[1].seed), (5, 11, 12));
        assert_eq!(c[14].regime.capacity, L4.capacity(&QWEN, 4.0));
    }

    #[test]
    fn counts_get_thousands_separators() {
        let t = [0, 999, 1_000, 36_400, 1_234_567].map(thousands);
        assert_eq!(t, ["0", "999", "1,000", "36,400", "1,234,567"]);
        let t = [189.75, 47.4375, 121.0, 1_979.0, 999.96].map(tenths);
        assert_eq!(t, ["189.8", "47.4", "121.0", "1,979.0", "1,000.0"]);
        let t = [16 << 10, 23_040, 80 << 10, 1_200 << 10, 1_536].map(gib);
        assert_eq!(t, ["16", "22.5", "80", "1,200", "1.5"]);
    }

    /// A placement whose arms have resume TTFT p50 100 ms and p90 1,000 ms (`oracle` for
    /// `oracle-prefetch`), KV pressure `pressure` and a realtime tier blocked `blocked` of the time.
    fn placement(gpu: Gpu, seed: u64, oracle: f64, pressure: f64, blocked: f64) -> Placement {
        let arms = Arm::E1
            .iter()
            .map(|&arm| Summary {
                arm: arm.name().into(),
                resume_ttft_ms: Percentiles {
                    p50: 100.0,
                    p90: if arm == Arm::OraclePrefetch {
                        oracle
                    } else {
                        1_000.0
                    },
                    ..Percentiles::default()
                },
                regime: Regime {
                    kv_pressure_mean: pressure,
                    kv_pressure_peak: 2.0 * pressure,
                    median_prompt_tokens: 4_000.0,
                    ..Regime::default()
                },
                replay: ReplayStats {
                    realtime_short_share: Some(1.0),
                    realtime_blocked_share: Some(blocked),
                    ..ReplayStats::default()
                },
                ..Summary::default()
            })
            .collect();
        Placement {
            gpu,
            token_scale: 8,
            result: regime::CellResult {
                cell: regime::Cell {
                    pauses: Pauses::Trace,
                    sessions: 16,
                    cpu_ratio: 4.0,
                    capacity: gpu.capacity(&QWEN, 4.0),
                },
                seed,
                arms,
            },
        }
    }

    #[test]
    fn the_tables_have_a_row_per_cell_and_arm_and_a_mean_per_cell() {
        let results = [
            placement(H100_SXM, 11, 700.0, 1.0, 0.1),
            placement(H100_SXM, 12, 900.0, 2.0, 0.3),
        ];
        let table = csv(&results);
        let columns = table.lines().next().unwrap().split(',').count();
        assert_eq!(table.lines().count(), 1 + 2 * Arm::E1.len());
        let row = table
            .lines()
            .find(|l| l.starts_with("h100-sxm,modeled,8,502432,12,trace,16,4,oracle-prefetch,"))
            .expect("the second seed's oracle row");
        assert_eq!(row.split(',').count(), columns);
        let chunk = chunk_ms(&H100_SXM.step_cost(), 4_000);
        assert!(
            row.ends_with(&format!(",4000,1.0000,0.3000,{chunk:.1}")),
            "{row}"
        );
        // One row for both seeds, at the GPU's 4x (2-token hash blocks at token scale 8).
        let map = markdown(&results, 2, 2_048);
        assert!(
            map.contains("seeds 11, 12") && map.contains("less than 2,048 tokens"),
            "{map}"
        );
        let rows: Vec<_> = map.lines().filter(|l| l.starts_with("| H100")).collect();
        assert_eq!(rows.len(), 1, "{map}");
        assert!(
            rows[0].starts_with("| H100 SXM 80 GB (modeled), 4× | 16 | 1.5 / 3.0 |"),
            "{map}"
        );
        // Short of KV 100% of the time and blocked 20% on average; the oracle gains 0 / 20%.
        assert!(
            rows[0].contains("| 100 / 20% |") && rows[0].contains("| +0 / +20% |"),
            "{map}"
        );
        assert!(rows[0].contains("| 100 / 1,000 |"), "{map}");
        let gpus = profiles(&GPUS);
        assert!(
            gpus.contains("| RTX 5060 Ti 16 GB | measured | 16 GiB | 448 GB/s |"),
            "{gpus}"
        );
        assert!(
            gpus.contains("PCIe 4.0 x16, 31.5 GB/s | 83,728 tokens | 17.26 ms"),
            "{gpus}"
        );
        assert!(gpus.contains("| 1,979.0 / 989.5 TFLOPS |"), "{gpus}");
        assert!(gpus.contains("| 502,432 tokens | 1.55 ms |"), "{gpus}");
    }
}
