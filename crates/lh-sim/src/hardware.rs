//! Other GPUs on the E1 regime map, from modeled costs.
//!
//! Every cost `lh-sim` uses is measured on the frozen box, an RTX 5060 Ti 16 GB: E0's tier fits
//! ([`TierCost::frozen_box`]) and the step-time model ([`StepCost::frozen_box`]). A [`Gpu`] holds
//! another card's published peaks (memory, memory bandwidth, dense FP8 and BF16 tensor throughput,
//! the host link) and the path vLLM 0.32.0 runs the box's checkpoint's linear layers on there
//! ([`LinearPath`]), and the peaks are used only as ratios to the box's: each measured term is
//! divided by the ratio of the published peaks of the kind that bounds it, so the box's measured
//! efficiency carries over. Terms the host and the software set stay the box's: the API server's
//! time, the restore's fixed part, the mixed-step extra and the restore stall. A [`Host`] that
//! `benchmarks/a100/bench_host.py` measured places the card on that host instead: the API server's
//! time follows its CPU, and the restore's transfer its measured link in place of the card's
//! datasheet one. The box's own profile gives its measured costs back exactly; every other GPU's
//! are modeled, and labeled so. A [`Model`] scales the same terms by another model's bytes and
//! FLOPs, and a [`Scaling`] says what carries over beyond the peaks: the E1 map's
//! ([`Scaling::MAP`]) or a forecast's range.
//!
//! The pool is the GPU's memory, less what its default ECC keeps and what the box keeps beside its
//! own (the engine, STT and TTS on one card: the edge pattern), in whole blocks, with the CPU tier a
//! multiple of it. [`sweep`] runs the E1 arms in each GPU's cells and records where it sits on the
//! map (KV pressure, π, ρ), how long the agents leave a realtime tier short of KV, and the step a
//! realtime turn waits behind.

use std::fmt::Write as _;
use std::path::Path;
use std::str::FromStr;

use lh_core::{
    FROZEN_BOX_POOL_TOKENS, KvGeometry, Linear, PromptSources, Quadratic, StepCost, StepLoad,
    TierCapacity, TierCost,
};
use serde::Deserialize;

use crate::aiperf::PlanOptions;
use crate::corpus::Corpus;
use crate::metrics::Summary;
use crate::regime::{self, signed, thousands};
use crate::replay::{Pauses, ReplayConfig, Service};
use crate::run::{Arm, RunConfig};
use crate::scheduler::StepBudget;

/// A PCIe host link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub generation: u8,
    pub lanes: u8,
}

impl Link {
    /// The rate in each direction, in GB/s: 8, 16 or 32 GT/s a lane for PCIe 3.0, 4.0 and 5.0, with
    /// 128b/130b encoding.
    pub fn gb_per_s(self) -> f64 {
        let gt_per_s = match self.generation {
            3 => 8.0,
            4 => 16.0,
            5 => 32.0,
            g => panic!("PCIe {g}.0 is not modeled"),
        };
        gt_per_s * f64::from(self.lanes) * 128.0 / 130.0 / 8.0
    }
}

/// How vLLM 0.32.0 runs the linear layers of a block-FP8 checkpoint (Qwen3's FP8 checkpoints quantize
/// their weights in 128x128 blocks with dynamic activations). For such a checkpoint
/// `Fp8LinearMethod` asks for a kernel with a per-group activation key, which takes the block list
/// `_POSSIBLE_FP8_BLOCK_KERNELS` (`model_executor/layers/quantization/fp8.py:258, 347`;
/// `model_executor/kernels/linear/__init__.py:743-755`), and the first kernel in it that supports
/// the card serves every linear layer (`__init__.py:478-488, 630-652, 712`). The per-tensor list is
/// another one and is not this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearPath {
    /// W8A8 at the FP8 tensor rate: DeepGEMM on sm90 and on the sm100 and sm120 families
    /// (`platforms/cuda.py:689-695`), the box's `DeepGemmFp8BlockScaledMMKernel` (its engine log), or
    /// CUTLASS's block kernel from sm90 (`csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu:161-174`).
    Fp8,
    /// FP8 weights through Marlin with BF16 activations (W8A16), at the BF16 tensor rate. On sm75 to
    /// sm89 the kernels ahead of `MarlinFP8ScaledMMLinearKernel` in the block list refuse the card
    /// (FlashInfer's needs DeepGEMM as its fallback, DeepGEMM sm90 or newer, CUTLASS sm90 or newer,
    /// B12x the sm120 family: `scaled_mm/BlockScaledMMLinearKernel.py:173-190`,
    /// `scaled_mm/deep_gemm.py:47-52`, `scaled_mm/cutlass.py:290-297`, `scaled_mm/b12x.py:53-67`),
    /// and Marlin takes any card from sm75 (`quantization/utils/marlin_utils_fp8.py:30-31`). Its
    /// FP8 path is weight-only: it raises on an 8-bit input (`marlin_utils_fp8.py:78-80`).
    Marlin,
}

impl LinearPath {
    pub fn label(self) -> &'static str {
        match self {
            LinearPath::Fp8 => "FP8 W8A8",
            LinearPath::Marlin => "Marlin W8A16",
        }
    }
}

/// Marlin's time over a BF16 GEMM's at the step cap's 512 tokens: vLLM PR #61224's table for
/// Qwen3-8B's four linear layers on an L4, the sums of their medians in µs (qkv 534.9 against 415.1,
/// o 298.8 against 229.4, gate-up 3,043.6 against 2,021.8, down 1,169.6 against 897.4;
/// <https://github.com/vllm-project/vllm/pull/61224>), measured. The only published figure for
/// Marlin's FP8 path at these sizes: the map uses it for every card that runs Marlin.
pub const MARLIN_FACTOR: f64 =
    (534.9 + 298.8 + 3_043.6 + 1_169.6) / (415.1 + 229.4 + 2_021.8 + 897.4);

/// A GPU, by its published peaks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gpu {
    /// The name on the command line.
    pub name: &'static str,
    pub title: &'static str,
    /// Memory in MiB, less what the card's default ECC keeps of it.
    pub memory_mib: u64,
    pub bandwidth_gb_s: f64,
    /// Dense tensor throughput, in TFLOPS; 0 for a card without FP8.
    pub fp8_tflops: f64,
    pub bf16_tflops: f64,
    pub link: Link,
    /// The path the box's checkpoint's linear layers take on the card.
    pub linear: LinearPath,
    /// FlashAttention 2's published forward efficiency on the card, at its best and its worst, where
    /// one is published; the forecast's range uses it.
    pub fa2_efficiency: Option<(f64, f64)>,
}

/// The frozen box. NVIDIA gives 16 GB of GDDR7 at 448 GB/s and 759 AI TOPS, FP4 with sparsity
/// (<https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/>,
/// <https://www.nvidia.com/en-us/geforce/graphics-cards/compare/>): 379.5 dense FP4 TFLOPS. The 5060
/// Ti is not in the RTX Blackwell whitepaper's tables (appendices A to C,
/// <https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf>),
/// so its rates follow from its FP4 rate by the ratios every GeForce Blackwell card there has: half
/// its FP4 rate in FP8 with FP16 accumulate, and a quarter of that in BF16, whose only rate is with
/// FP32 accumulate. FP8 with FP32 accumulate runs at half rate on these cards, and E0's recompute
/// cell says the box's linear layers beat that: at 4,096 tokens, 437 ms less the 48 ms fixed part
/// leaves 389 ms, where the linear layers alone at the FP32-accumulate peak (94.9 TFLOPS) would take
/// 314 ms and attention at 47.4 TFLOPS about 104 ms more, 418 ms. At the FP16-accumulate peak they
/// take 157 ms, which fits, so that is the peak of the same kind. The link is what both hosts
/// negotiate (`nvidia-smi`'s `pcie.link.gen.hostmax` 3, width 8), not the card's PCIe 5.0.
pub const RTX_5060_TI: Gpu = Gpu {
    name: "rtx-5060-ti",
    title: "RTX 5060 Ti 16 GB",
    memory_mib: 16 << 10,
    bandwidth_gb_s: 448.0,
    fp8_tflops: 759.0 / 4.0,
    bf16_tflops: 759.0 / 16.0,
    link: Link {
        generation: 3,
        lanes: 8,
    },
    linear: LinearPath::Fp8,
    fa2_efficiency: None,
};

/// NVIDIA's L4 datasheet (<https://www.nvidia.com/en-us/data-center/l4/>): 24 GB at 300 GB/s, 485 FP8
/// and 242 BF16 TFLOPS "shown with sparsity", halved, and PCIe Gen4 x16. Its product brief
/// (PB-11316-001, tables 2 and 3,
/// <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf>) gives
/// the 24 GB as GDDR6 with ECC "Enabled (by default)", and with ECC on, GDDR memory keeps its check
/// bits in 6.25% of itself (the CUDA C++ Best Practices Guide, 9.2.1,
/// <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#theoretical-bandwidth-calculation>).
/// That leaves 22.5 GiB, of which `nvidia-smi` shows 23,034 MiB
/// (<https://github.com/NVIDIA/TensorRT-LLM/issues/1823>), as it shows 16,311 MiB of the box's
/// 16 GiB. An L4 (sm89) runs the box's block-FP8 checkpoint through Marlin ([`LinearPath::Marlin`]),
/// as PR #61224 observed on one ("vLLM logs 'Your GPU does not have native support for FP8
/// computation'"), so its FP8 rate goes unused.
pub const L4: Gpu = Gpu {
    name: "l4",
    title: "L4 24 GB",
    memory_mib: (24 << 10) * 15 / 16,
    bandwidth_gb_s: 300.0,
    fp8_tflops: 485.0 / 2.0,
    bf16_tflops: 242.0 / 2.0,
    link: Link {
        generation: 4,
        lanes: 16,
    },
    linear: LinearPath::Marlin,
    fa2_efficiency: None,
};

/// NVIDIA's A100 datasheet (JUN21,
/// <https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>):
/// 40 GB of HBM2 at 1,555 GB/s, 312 BF16 TFLOPS dense (624 with sparsity) and no FP8, and PCIe Gen4
/// x16 at 64 GB/s both ways; the SXM4 and PCIe cards differ in their power limit (400 and 250 W) and
/// NVLink, not in these. HBM keeps ECC in bits of its own: an A100-SXM4-40GB's `nvidia-smi` shows
/// 40,960 MiB
/// (<https://forums.developer.nvidia.com/t/a100-sxm4-40gb-shows-in-nvidia-smi-but-not-accessible-in-cuda/322730>).
/// sm80 runs the box's checkpoint through Marlin ([`LinearPath::Marlin`]). FlashAttention 2's
/// forward pass reaches "50-73% of the theoretical maximum FLOPs/s on A100"
/// (<https://arxiv.org/abs/2307.08691>), against the 88% of its BF16 peak the box's attention term
/// carries.
pub const A100_40GB: Gpu = Gpu {
    name: "a100-40gb",
    title: "A100 40 GB",
    memory_mib: 40 << 10,
    bandwidth_gb_s: 1_555.0,
    fp8_tflops: 0.0,
    bf16_tflops: 312.0,
    link: Link {
        generation: 4,
        lanes: 16,
    },
    linear: LinearPath::Marlin,
    fa2_efficiency: Some((0.73, 0.50)),
};

/// NVIDIA's A100 datasheet (JUN21, the 40 GB card's source,
/// <https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>),
/// its "A100 80GB SXM" column: 80 GB of HBM2e at 2,039 GB/s (the 80 GB PCIe card's is 1,935), and
/// the 40 GB cards' 312 BF16 TFLOPS dense (624 with sparsity), no FP8 and PCIe Gen4 at 64 GB/s. It
/// is sm80, so Marlin and FlashAttention 2's range on the A100 carry over: its profile is the 40 GB
/// card's with the memory and the bandwidth changed, and every cost follows from them by the 40 GB
/// card's rules. HBM2e keeps ECC in bits of its own: an A100-SXM4-80GB's `nvidia-smi` shows
/// 81,920 MiB (<https://github.com/NVIDIA/DCGM/issues/93>). The A100 run drew this card; the E1 map
/// does not place it ([`GPUS`]).
pub const A100_80GB: Gpu = Gpu {
    name: "a100-80gb",
    title: "A100 SXM4 80 GB",
    memory_mib: 80 << 10,
    bandwidth_gb_s: 2_039.0,
    ..A100_40GB
};

/// NVIDIA's H100 datasheet, SXM (<https://www.nvidia.com/en-us/data-center/h100/>): 80 GB at 3.35 TB/s,
/// 3,958 FP8 and 1,979 BF16 TFLOPS with sparsity, halved, and PCIe Gen5 at 128 GB/s (x16, both ways).
pub const H100_SXM: Gpu = Gpu {
    name: "h100-sxm",
    title: "H100 SXM 80 GB",
    memory_mib: 80 << 10,
    bandwidth_gb_s: 3_350.0,
    fp8_tflops: 3_958.0 / 2.0,
    bf16_tflops: 1_979.0 / 2.0,
    link: Link {
        generation: 5,
        lanes: 16,
    },
    linear: LinearPath::Fp8,
    fa2_efficiency: None,
};

/// The profiles the E1 map places (`lh-sim hardware`'s default), the box first.
pub const GPUS: [Gpu; 4] = [RTX_5060_TI, L4, A100_40GB, H100_SXM];

/// Every profile a command takes by name: the map's, and the A100 80 GB SXM4, which `forecast` models
/// for the run on it (`benchmarks/a100/forecast-80gb.md`).
pub const CARDS: [Gpu; 5] = [RTX_5060_TI, L4, A100_40GB, A100_80GB, H100_SXM];

/// A model as the cost model scales it: what a decode step reads, what a computed token and an
/// attention pair compute, and its KV. The box's measured terms are for [`QWEN3_4B_FP8`]; another
/// model's are those terms times its ratios to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Model {
    pub name: &'static str,
    pub title: &'static str,
    /// Bytes of weights resident on the GPU.
    pub weight_bytes: f64,
    /// Bytes of weights a decode step reads: every linear layer and the LM head.
    pub step_bytes: f64,
    /// Linear-layer FLOPs a computed token takes: twice the linear weights.
    pub token_flops: f64,
    /// Attention FLOPs a pair takes (one query token against one cached token): 4 × head dim ×
    /// query heads × layers.
    pub pair_flops: f64,
    pub geometry: KvGeometry,
}

/// The box's model (`config.json`, <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8>): 36
/// layers of 100.9 M FP8 linear weights (q 2,560×4,096, k and v 2,560×1,024 each, o 4,096×2,560, gate
/// and up 2,560×9,728 each, down 9,728×2,560), 3.63 GB, and the 0.78 GB BF16 embedding (151,936 ×
/// 2,560), tied to the LM head, which every step reads; 32 query heads of 128.
pub const QWEN3_4B_FP8: Model = Model {
    name: "qwen3-4b-fp8",
    title: "Qwen3-4B-Instruct-2507-FP8",
    weight_bytes: 4.41e9,
    step_bytes: 4.41e9,
    token_flops: 7.27e9,
    pair_flops: 4.0 * 128.0 * 32.0 * 36.0,
    geometry: KvGeometry::QWEN3_4B_BF16,
};

/// Qwen3-14B-FP8 (`config.json`, <https://huggingface.co/Qwen/Qwen3-14B-FP8>, block-FP8 like the
/// box's): 40 layers of 330.3 M FP8 linear weights (q and o 5,120×5,120 each, k and v 5,120×1,024
/// each, gate and up 5,120×17,408 each, down 17,408×5,120), 13.21 GB, and an embedding and an LM head
/// that are not tied, 1.56 GB of BF16 each (151,936 × 5,120); a step reads the linear layers and the
/// LM head. 40 query heads and 8 KV heads of 128: 160 KiB of BF16 KV a token.
pub const QWEN3_14B_FP8: Model = Model {
    name: "qwen3-14b-fp8",
    title: "Qwen3-14B-FP8",
    weight_bytes: 13.21e9 + 2.0 * 1.556e9,
    step_bytes: 13.21e9 + 1.556e9,
    token_flops: 2.0 * 13.21e9,
    pair_flops: 4.0 * 128.0 * 40.0 * 40.0,
    geometry: KvGeometry::new(40, 8, 128, 2, 16),
};

/// The models `forecast` sizes.
pub const MODELS: [Model; 2] = [QWEN3_4B_FP8, QWEN3_14B_FP8];

impl FromStr for Model {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        MODELS.into_iter().find(|m| m.name == s).ok_or_else(|| {
            let names: Vec<_> = MODELS.iter().map(|m| m.name).collect();
            format!("unknown model {s:?} (the models are {})", names.join(", "))
        })
    }
}

/// The share of the box's fixed step term that is the weight read: 4.41 GB at 448 GB/s, 9.84 of
/// the 11.56 ms. The rest, 1.72 ms, is launches and kernel tails.
const BOX_READ_SHARE: f64 =
    QWEN3_4B_FP8.step_bytes / (RTX_5060_TI.bandwidth_gb_s * 1e6) / StepCost::frozen_box().step_ms;

/// The share of its dense BF16 peak the box's attention term achieves: 589,824 FLOPs a pair in
/// 14.17 ns is 41.6 TFLOPS, 88% of 47.4.
pub const BOX_ATTENTION_EFFICIENCY: f64 = QWEN3_4B_FP8.pair_flops
    / (RTX_5060_TI.bf16_tflops * 1e12 * StepCost::frozen_box().ns_per_pair * 1e-9);

/// What a modeled card's costs carry from the box beyond the ratios of their peaks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scaling {
    /// Keep the box's fixed-step time beyond its weight read as it is, instead of scaling it by
    /// memory bandwidth with the read.
    pub keep_step_overhead: bool,
    /// On a card that runs Marlin, its time over a BF16 GEMM's.
    pub marlin_factor: f64,
    /// Attention's share of the card's dense BF16 peak; `None` carries the box's.
    pub attention_efficiency: Option<f64>,
}

impl Scaling {
    /// The E1 map's, on every card: the fixed step scaled by bandwidth with the read (a floor), Marlin
    /// as an L4 measured it ([`MARLIN_FACTOR`]), and the box's attention efficiency.
    pub const MAP: Self = Self {
        keep_step_overhead: false,
        marlin_factor: MARLIN_FACTOR,
        attention_efficiency: None,
    };

    /// The fastest a forecast allows: the peaks' ratios alone, with Marlin as fast as a BF16 GEMM.
    pub const LOW: Self = Self {
        keep_step_overhead: false,
        marlin_factor: 1.0,
        attention_efficiency: None,
    };

    /// A forecast's central costs for `gpu`: the step's overhead kept, as vLLM measured it on an H100
    /// (the README), Marlin as on the L4, and attention at FlashAttention 2's best published
    /// efficiency on the card, where there is one.
    pub fn central(gpu: &Gpu) -> Self {
        Self {
            keep_step_overhead: true,
            marlin_factor: MARLIN_FACTOR,
            attention_efficiency: gpu.fa2_efficiency.map(|(best, _)| best),
        }
    }

    /// The slowest a forecast allows: as [`central`](Self::central), with attention at FlashAttention
    /// 2's worst published efficiency.
    pub fn high(gpu: &Gpu) -> Self {
        Self {
            attention_efficiency: gpu.fa2_efficiency.map(|(_, worst)| worst),
            ..Self::central(gpu)
        }
    }
}

/// The box's host at `benchmarks/a100/bench_host.py`'s Python workload, in ms: 381.1 at the median
/// of seven on Phastform's Xeon E5-2609 v4 at 1.70 GHz (2026-10-26, its GPU units stopped), as its
/// `host.json` gives it (`testdata/box-host.json` is a copy), so that the box's own file places a
/// card on the box's CPU exactly.
pub const BOX_PYTHON_MS: f64 = 381.100_237_369_537_35;

/// The box's host from pinned host memory to the card in one copy, in GB/s, in the same `host.json`:
/// 268,435,456 bytes in 39.69 ms, 6.76 GB/s of the 7.88 its PCIe 3.0 x8 link gives.
pub const BOX_H2D_GB_PER_S: f64 = 6.763_364_628_525_3;

/// The share of its host's one-copy rate the box's E0 restore reaches: 147,456 bytes of KV a token in
/// [`TierCost::frozen_box`]'s 23.64 µs is 6.24 GB/s, 92% of [`BOX_H2D_GB_PER_S`]. (E0's seven p50
/// cells give 23.60 µs a token by least squares, as the A100 analysis reads them: 6.25 GB/s.)
pub const BOX_RESTORE_SHARE: f64 = QWEN3_4B_FP8.geometry.bytes_per_token as f64
    / (TierCost::frozen_box().cpu_restore.us_per_token * 1e3)
    / BOX_H2D_GB_PER_S;

/// A host as `benchmarks/a100/bench_host.py` measures it, from its `host.json`: the CPU at a fixed
/// Python workload of the kind vLLM's API server and scheduler run, and the link from pinned host
/// memory to the card in one copy. The A100 run showed both to be the host's, not the card's: its
/// host's API server took 14.2 ms + 4.971 µs a prompt token where the box's takes 29.6 ms + 7.994,
/// and its restores moved 11.46 GB/s of KV where the forecast took 25 from the card's PCIe Gen4 x16
/// link, through a CPU with PCIe 3.0 root ports. A card placed on one ([`Gpu::step_cost_on`],
/// [`Gpu::tier_cost_on`]) takes the API server's time from its CPU and the restore's transfer from
/// its link.
#[derive(Clone, Debug, PartialEq)]
pub struct Host {
    /// The CPU's model name, where the file gives it.
    pub cpu: Option<String>,
    /// The workload's median, in ms.
    pub python_ms: f64,
    /// Pinned host memory to the card in one copy, in GB/s.
    pub h2d_gb_per_s: f64,
}

// What a forecast reads of a host.json.
#[derive(Deserialize)]
struct BenchHost {
    cpu: Option<String>,
    python_ms: Option<Workload>,
    #[serde(default)]
    link: Vec<LinkRow>,
}

#[derive(Deserialize)]
struct Workload {
    p50: Option<f64>,
}

#[derive(Deserialize)]
struct LinkRow {
    direction: String,
    /// `None`: the bytes as one copy.
    page_bytes: Option<u64>,
    gb_per_s: Option<f64>,
}

impl Host {
    /// The host in `json`, the `host.json` bench_host.py writes: its CPU's name, the workload's
    /// median (`python_ms.p50`) and the link's rate host to device in one copy (the `h2d` row whose
    /// `page_bytes` is null). Without the median or that row it is refused: bench_host.py measures
    /// the link only on a GPU (not with `--no-gpu`), and writes no row when the measurement fails.
    pub fn from_bench_host(json: &str) -> Result<Self, String> {
        let file: BenchHost = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let python_ms = file
            .python_ms
            .and_then(|w| w.p50)
            .ok_or("python_ms.p50 is missing")?;
        let one_copy = file
            .link
            .iter()
            .find(|row| row.direction == "h2d" && row.page_bytes.is_none())
            .ok_or("the link's h2d row of one copy (page_bytes null) is missing")?;
        let h2d_gb_per_s = one_copy
            .gb_per_s
            .ok_or("the link's h2d row of one copy has no gb_per_s")?;
        for (name, x) in [
            ("python_ms.p50", python_ms),
            ("the h2d row's gb_per_s", h2d_gb_per_s),
        ] {
            if x <= 0.0 {
                return Err(format!("{name} is {x}, not a positive number"));
            }
        }
        Ok(Self {
            cpu: file.cpu,
            python_ms,
            h2d_gb_per_s,
        })
    }

    /// How many times faster than the box's this host's CPU runs the workload: the API server's time
    /// on it is the box's over this.
    pub fn cpu_ratio(&self) -> f64 {
        BOX_PYTHON_MS / self.python_ms
    }

    /// This host's one-copy rate over the box's: a restore's transfer on it is the box's over this.
    pub fn link_ratio(&self) -> f64 {
        self.h2d_gb_per_s / BOX_H2D_GB_PER_S
    }
}

/// Qwen3-4B-Instruct-2507's native context
/// (<https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8>): vLLM serves no longer request of it.
pub const MODEL_CONTEXT: u64 = 262_144;

/// The KV a realtime tier keeps per voice call: a turn's prompt and reply, about 680 tokens in the
/// W2 runs on the box, rounded up to 64 blocks.
pub const TOKENS_PER_CALL: u64 = 1_024;

/// What the box keeps beside its pool, of its 16 GiB: 9,288 MiB of engine, STT and TTS (14,408 MiB in
/// use with the three servers idle, `docs/BENCHMARKS.md`, less the engine's 5 GiB `kv-cache-memory`:
/// weights, activations, CUDA graphs and contexts) and 1,976 MiB of headroom (`nvidia-smi` shows
/// 16,311 MiB, 1,903 MiB of them free).
const BOX_BESIDE_POOL: u64 = (16 << 30) - (5 << 30);

/// The regime map's largest session count, where the box is at pressure 7 to 7.5.
const BOX_TOP_SESSIONS: f64 = 40.0;

/// A GPU's peaks over the box's, as a [`Scaling`] weighs them, and its host's over the box's host's.
struct Ratios {
    bandwidth: f64,
    /// The rate the linear layers run at over the box's FP8 rate.
    linear: f64,
    /// Attention's achieved rate over the box's.
    attention: f64,
    /// The host's measured one-copy rate over the box's ([`Host::link_ratio`]), or without a host
    /// the card's datasheet link over the box's.
    link: f64,
    /// How many times faster than the box's the host's CPU runs Python: 1 without a host.
    cpu: f64,
}

/// A model's terms over the box's model's.
struct ModelRatios {
    step_bytes: f64,
    token_flops: f64,
    pair_flops: f64,
    kv_bytes: f64,
}

impl Model {
    fn over_box(&self) -> ModelRatios {
        let b = &QWEN3_4B_FP8;
        ModelRatios {
            step_bytes: self.step_bytes / b.step_bytes,
            token_flops: self.token_flops / b.token_flops,
            pair_flops: self.pair_flops / b.pair_flops,
            kv_bytes: self.geometry.bytes_per_token as f64 / b.geometry.bytes_per_token as f64,
        }
    }
}

impl Gpu {
    /// Whether these are the box's measured costs rather than modeled ones.
    pub fn measured(&self) -> bool {
        self.name == RTX_5060_TI.name
    }

    pub fn label(&self) -> &'static str {
        if self.measured() {
            "measured"
        } else {
            "modeled"
        }
    }

    fn ratios(&self, scaling: Scaling, host: Option<&Host>) -> Ratios {
        let b = &RTX_5060_TI;
        // The box's linear layers run FP8; Marlin runs at the BF16 rate, `marlin_factor` times slower.
        let linear = match self.linear {
            LinearPath::Fp8 => self.fp8_tflops,
            LinearPath::Marlin => self.bf16_tflops / scaling.marlin_factor,
        };
        let efficiency = scaling
            .attention_efficiency
            .map_or(1.0, |e| e / BOX_ATTENTION_EFFICIENCY);
        Ratios {
            bandwidth: self.bandwidth_gb_s / b.bandwidth_gb_s,
            linear: linear / b.fp8_tflops,
            attention: self.bf16_tflops / b.bf16_tflops * efficiency,
            link: host.map_or_else(
                || self.link.gb_per_s() / b.link.gb_per_s(),
                Host::link_ratio,
            ),
            cpu: host.map_or(1.0, Host::cpu_ratio),
        }
    }

    /// The fixed step term over the box's: the weight read scaled by the model's bytes and the card's
    /// bandwidth, and the box's 1.72 ms beyond it kept or scaled with it. Written so that the box's
    /// model under [`Scaling::MAP`] gives `1 / bandwidth ratio` exactly.
    fn fixed_step_over_box(&self, model: &Model, scaling: Scaling) -> f64 {
        let bandwidth = self.bandwidth_gb_s / RTX_5060_TI.bandwidth_gb_s;
        let read = model.over_box().step_bytes;
        if scaling.keep_step_overhead {
            BOX_READ_SHARE * read / bandwidth + (1.0 - BOX_READ_SHARE)
        } else {
            (1.0 + BOX_READ_SHARE * (read - 1.0)) / bandwidth
        }
    }

    /// The step-time model: the box's fit with its weight and KV reads scaled by memory bandwidth,
    /// its linear layers by the rate of the card's [`LinearPath`] and its attention by BF16
    /// throughput, under the E1 map's [`Scaling`], for the box's model.
    pub fn step_cost(&self) -> StepCost {
        self.step_cost_with(&QWEN3_4B_FP8, Scaling::MAP)
    }

    /// The step-time model for `model` under `scaling`, on the box's CPU.
    pub fn step_cost_with(&self, model: &Model, scaling: Scaling) -> StepCost {
        self.step_cost_on(model, scaling, None)
    }

    /// The step-time model for `model` under `scaling` with the card on `host`, whose CPU sets the
    /// API server's time; on the box's CPU without one.
    pub fn step_cost_on(&self, model: &Model, scaling: Scaling, host: Option<&Host>) -> StepCost {
        let r = self.ratios(scaling, host);
        let m = model.over_box();
        let base = StepCost::frozen_box();
        StepCost {
            // Mostly the weight read: Qwen3-4B-FP8 reads 4.41 GB a step (3.63 G FP8 linear weights
            // and the 0.78 GB BF16 embedding, tied to the LM head), 9.8 ms of the 11.56 ms at the
            // box's 448 GB/s. The other 1.7 ms, launches and kernel tails, need not shrink with
            // bandwidth: vLLM 0.30 decoding Qwen3-4B in BF16 at batch 1 on an H100 SXM takes 4.29 ms
            // a step, 2.45 ms of it the weight read at the datasheet's bandwidth and 1.8 ms the rest
            // (decode-decoded, <https://github.com/zhebrak/decode-decoded>, results/cells.csv). So
            // the H100's 1.55 ms under the map's scaling is a floor: with the box's 1.7 ms kept
            // (`keep_step_overhead`) it is 3.0 ms.
            step_ms: base.step_ms * self.fixed_step_over_box(model, scaling),
            // The linear layers.
            us_per_token: base.us_per_token * m.token_flops / r.linear,
            // Attention over the BF16 cache.
            ns_per_pair: base.ns_per_pair * m.pair_flops / r.attention,
            // The KV reads.
            ns_per_context: base.ns_per_context * m.kv_bytes / r.bandwidth,
            // The API server's time, Python on the host's CPU, over how many times faster that CPU
            // runs bench_host.py's workload: its work on a prompt (`host`: rendering, tokenizing,
            // hashing its blocks) and the extra for each request decoding when one arrives
            // (`host_per_decode_ms`: lh-core's extra host time for the requests whose tokens it
            // streams meanwhile, which fit_step_cost.py fits as GPU hits' excess over the host time
            // and their steps, by how many requests decode).
            host: faster(base.host, r.cpu),
            host_per_decode_ms: base.host_per_decode_ms / r.cpu,
            // The other two are not host time, and stay the software's. The mixed-step extra is a
            // step's: the median excess over the step terms of decode steps beside a prefill
            // (fit_step_cost.py), the engine's forward pass, whose CUDA graphs and kernels the
            // software sets. The restore stall is the share of a restore's transfer the step loop
            // waits (lost decode time over E0's transfer time), as the connector overlaps a copy
            // with the steps; its time follows the transfer, which the host's link sets. On the A100
            // run's host, whose API server took half the box's time, fit_step_cost.py gave 7.3 ms
            // and 0.82 against the box's 7.6 and 0.80.
            ..base
        }
    }

    /// The tier costs under the E1 map's [`Scaling`], for the box's model: the restore's transfer
    /// scaled by the host link, and E0's cold prefill by the parts of the step loop that make it up.
    pub fn tier_cost(&self) -> TierCost {
        self.tier_cost_with(&QWEN3_4B_FP8, Scaling::MAP)
    }

    /// The tier costs for `model` under `scaling`, on the box's CPU and the card's datasheet link.
    pub fn tier_cost_with(&self, model: &Model, scaling: Scaling) -> TierCost {
        self.tier_cost_on(model, scaling, None)
    }

    /// The tier costs for `model` under `scaling` with the card on `host`: the restore's transfer by
    /// the host's measured link, and E0's cold prefill with E0's host time on its CPU. Without one,
    /// the box's CPU and the card's datasheet link.
    pub fn tier_cost_on(&self, model: &Model, scaling: Scaling, host: Option<&Host>) -> TierCost {
        let r = self.ratios(scaling, host);
        let m = model.over_box();
        let base = TierCost::frozen_box();
        let steps = self.step_cost_on(model, scaling, host);
        // E0's cold prefill: the box's measured fit at each prompt length, times how much longer the
        // scaled step loop takes a lone prefill of it than the box's does (with E0's host time, on
        // the card's host and on the box's), refitted as a quadratic over E0's lengths and the
        // longer prompts the map replays. So the fit keeps, on every card, the relation it has to
        // the box's own step loop (2% below it at 4k and 8k tokens, 14% above at 1k), within the
        // refit's 5%. A card whose step model is the box's, its host time included, gets the box's
        // fit back.
        let recompute = if steps == StepCost::frozen_box() {
            base.recompute
        } else {
            let lone = |s: &StepCost, host, tokens| lone_prefill_ms(s, host, &base, tokens);
            let points = RECOMPUTE_LENGTHS.map(|tokens| {
                let ratio =
                    lone(&steps, host, tokens) / lone(&StepCost::frozen_box(), None, tokens);
                (tokens, base.recompute_ms(tokens) * ratio)
            });
            fit_quadratic(&points)
        };
        TierCost {
            // The transfer by the link; the fixed part is the connector's. Gen4 x16 gives 5.91 µs a
            // token, 34.5, 58.7 and 107.1 ms for 4k, 8k and 16k tokens, where vLLM 0.29 measured 34,
            // 65 and 104 ms for Qwen3-4B on an L20, also PCIe Gen4
            // (<https://github.com/vllm-project/vllm/issues/57103#issuecomment-5733975478>). On a
            // measured host the link is its one-copy rate over the box's: a token's KV at that rate
            // times the share of its own the box's E0 restore reaches ([`BOX_RESTORE_SHARE`]).
            cpu_restore: Linear {
                ms: base.cpu_restore.ms,
                us_per_token: base.cpu_restore.us_per_token * m.kv_bytes / r.link,
            },
            recompute,
            // The GPU-hit fit and the decode rate are the idle service's, which the step loop
            // replaces.
            ..base
        }
    }

    /// The GPU pool in tokens: whole blocks of what the box would leave to its pool on this card.
    pub fn pool_tokens(&self, geometry: &KvGeometry) -> u64 {
        let bytes = (self.memory_mib << 20).saturating_sub(BOX_BESIDE_POOL);
        geometry.blocks_in_bytes(bytes) * geometry.block_tokens
    }

    /// The pool of `model` on this card, in tokens: what the box keeps beside its pool, with
    /// `model`'s weights in place of the box's model's, left to `model`'s KV in whole blocks. The
    /// box's model gets [`pool_tokens`](Self::pool_tokens) back.
    pub fn model_pool_tokens(&self, model: &Model) -> u64 {
        let beside = BOX_BESIDE_POOL as f64 + model.weight_bytes - QWEN3_4B_FP8.weight_bytes;
        let bytes = ((self.memory_mib << 20) as f64 - beside).max(0.0) as u64;
        model.geometry.blocks_in_bytes(bytes) * model.geometry.block_tokens
    }

    /// The pool and a CPU tier `cpu_ratio` times its size.
    pub fn capacity(&self, geometry: &KvGeometry, cpu_ratio: f64) -> TierCapacity {
        TierCapacity::with_cpu_ratio(geometry, self.pool_tokens(geometry), cpu_ratio)
    }

    /// The frozen box's run of `replay`, with this GPU's costs.
    pub fn run_config(&self, replay: ReplayConfig) -> RunConfig {
        RunConfig {
            cost: self.tier_cost(),
            service: Service::Contention {
                budget: StepBudget::FROZEN_BOX,
                steps: self.step_cost(),
            },
            ..RunConfig::frozen_box(replay)
        }
    }
}

/// The prompt lengths E0's cold-prefill fit is refitted over on a modeled card: E0's own, and the
/// longer prompts the map replays at the GPUs' own scalings.
const RECOMPUTE_LENGTHS: [u64; 10] = [
    1_024, 2_048, 4_096, 8_192, 16_384, 24_576, 31_744, 65_536, 131_072, 249_152,
];

/// E0's host time, its requests' (they carry token ids, so it is shorter than the chat requests':
/// [`StepCost::e0_host`]): the box's, and on `host` the box's over how many times faster its CPU
/// runs Python.
pub fn e0_host(host: Option<&Host>) -> Linear {
    let on_box = StepCost::frozen_box().e0_host(&TierCost::frozen_box());
    faster(on_box, host.map_or(1.0, Host::cpu_ratio))
}

/// `time` on a CPU `cpu` times as fast as the box's.
fn faster(time: Linear, cpu: f64) -> Linear {
    Linear {
        ms: time.ms / cpu,
        us_per_token: time.us_per_token / cpu,
    }
}

/// A cold prompt of `tokens` alone on the step loop `steps`, with E0's host time on `host`, in ms.
fn lone_prefill_ms(steps: &StepCost, host: Option<&Host>, tier: &TierCost, tokens: u64) -> f64 {
    let e0 = StepCost {
        host: e0_host(host),
        ..steps.clone()
    };
    let cold = PromptSources {
        computed: tokens,
        ..PromptSources::default()
    };
    e0.alone_ttft_ms(tier, cold, StepBudget::FROZEN_BOX.max_batched_tokens)
}

/// The quadratic in the token count through `points` (tokens, ms), by least squares of the relative
/// error.
fn fit_quadratic(points: &[(u64, f64)]) -> Quadratic {
    // In units of 10k tokens, to keep the normal equations well conditioned.
    let x = |tokens: u64| {
        let t = tokens as f64 / 1e4;
        [1.0, t, t * t]
    };
    let (mut a, mut b) = ([[0.0; 3]; 3], [0.0; 3]);
    for &(tokens, ms) in points {
        let (row, w) = (x(tokens), 1.0 / (ms * ms));
        for i in 0..3 {
            b[i] += w * row[i] * ms;
            for j in 0..3 {
                a[i][j] += w * row[i] * row[j];
            }
        }
    }
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    let solve = |k: usize| {
        let mut m = a;
        for (i, row) in m.iter_mut().enumerate() {
            row[k] = b[i];
        }
        det(m) / d
    };
    // c0 ms, c1 ms per 10k tokens, c2 ms per (10k tokens)².
    Quadratic {
        ms: solve(0),
        us_per_token: solve(1) / 10.0,
        ns_per_token_squared: solve(2) / 100.0,
    }
}

impl FromStr for Gpu {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        CARDS.into_iter().find(|g| g.name == s).ok_or_else(|| {
            let names: Vec<_> = CARDS.iter().map(|g| g.name).collect();
            format!("unknown GPU {s:?} (the GPUs are {})", names.join(", "))
        })
    }
}

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
    use lh_core::PromptSources;
    use serde_json::json;

    use super::*;
    use crate::metrics::{Percentiles, Regime};
    use crate::replay::ReplayStats;

    const QWEN: KvGeometry = KvGeometry::QWEN3_4B_BF16;

    /// The box with `f` applied to its peaks.
    fn the_box_with(f: impl FnOnce(&mut Gpu)) -> Gpu {
        let mut g = Gpu {
            name: "test",
            ..RTX_5060_TI
        };
        f(&mut g);
        g
    }

    #[test]
    fn the_box_gives_its_measured_costs_back() {
        let b = RTX_5060_TI;
        assert_eq!(b.step_cost(), StepCost::frozen_box());
        assert_eq!(b.tier_cost(), TierCost::frozen_box());
        assert_eq!(b.pool_tokens(&QWEN), FROZEN_BOX_POOL_TOKENS);
        assert_eq!(b.capacity(&QWEN, 4.0).gpu_blocks, 2_275);
        let replay = ReplayConfig::default();
        assert_eq!(b.run_config(replay.clone()), RunConfig::frozen_box(replay));
        assert!(b.measured() && !L4.measured() && !A100_40GB.measured() && !H100_SXM.measured());
        assert!(!A100_80GB.measured());
        assert_eq!((b.label(), H100_SXM.label()), ("measured", "modeled"));
        // Every scaling gives the box its own costs for its own model, except for the overhead kept,
        // which is the same term split in two.
        for s in [Scaling::MAP, Scaling::LOW, Scaling::high(&b)] {
            let steps = b.step_cost_with(&QWEN3_4B_FP8, s);
            assert!((steps.step_ms - StepCost::frozen_box().step_ms).abs() < 1e-12);
            assert_eq!(steps.us_per_token, StepCost::frozen_box().us_per_token);
            assert_eq!(steps.ns_per_pair, StepCost::frozen_box().ns_per_pair);
        }
        assert_eq!(
            b.tier_cost_with(&QWEN3_4B_FP8, Scaling::LOW),
            TierCost::frozen_box()
        );
        assert_eq!(b.model_pool_tokens(&QWEN3_4B_FP8), FROZEN_BOX_POOL_TOKENS);
        // 759 AI TOPS: 189.75 dense FP8 and 47.4 BF16 TFLOPS; PCIe 3.0 x8 is 7.88 GB/s each way.
        assert_eq!((b.fp8_tflops, b.bf16_tflops), (189.75, 47.4375));
        assert!((b.link.gb_per_s() - 7.877).abs() < 1e-3);
        assert!((H100_SXM.link.gb_per_s() / b.link.gb_per_s() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn each_peak_scales_only_its_terms() {
        let (box_steps, box_tier) = (StepCost::frozen_box(), TierCost::frozen_box());
        let software = |s: &StepCost, t: &TierCost| {
            assert_eq!(
                (s.mixed_ms, s.host, s.host_per_decode_ms, s.restore_stall),
                (
                    box_steps.mixed_ms,
                    box_steps.host,
                    box_steps.host_per_decode_ms,
                    box_steps.restore_stall
                )
            );
            assert_eq!(
                (
                    t.gpu_hit,
                    t.cpu_restore.ms,
                    t.inter_token_ms,
                    t.block_tokens
                ),
                (
                    box_tier.gpu_hit,
                    box_tier.cpu_restore.ms,
                    box_tier.inter_token_ms,
                    box_tier.block_tokens
                )
            );
        };
        // The cold prefill is cheaper at every length when a step term shrinks.
        let cheaper = |t: &TierCost| {
            for tokens in RECOMPUTE_LENGTHS {
                assert!(
                    t.recompute_ms(tokens) < box_tier.recompute_ms(tokens),
                    "{tokens}: {:.0} against {:.0}",
                    t.recompute_ms(tokens),
                    box_tier.recompute_ms(tokens)
                );
            }
        };
        // Twice the bandwidth: half the fixed step and the KV reads, and a cheaper cold prefill.
        let g = the_box_with(|g| g.bandwidth_gb_s *= 2.0);
        let (s, t) = (g.step_cost(), g.tier_cost());
        assert_eq!(s.step_ms, box_steps.step_ms / 2.0);
        assert_eq!(s.ns_per_context, box_steps.ns_per_context / 2.0);
        assert_eq!(
            (s.us_per_token, s.ns_per_pair, t.cpu_restore),
            (
                box_steps.us_per_token,
                box_steps.ns_per_pair,
                box_tier.cpu_restore
            )
        );
        cheaper(&t);
        software(&s, &t);
        // Twice the FP8 throughput: half the linear layers.
        let g = the_box_with(|g| g.fp8_tflops *= 2.0);
        let (s, t) = (g.step_cost(), g.tier_cost());
        assert_eq!(s.us_per_token, box_steps.us_per_token / 2.0);
        assert_eq!(
            (s.step_ms, s.ns_per_pair, s.ns_per_context),
            (
                box_steps.step_ms,
                box_steps.ns_per_pair,
                box_steps.ns_per_context
            )
        );
        cheaper(&t);
        software(&s, &t);
        // Twice the BF16 throughput: half the attention, the cold prefill's quadratic term most.
        let g = the_box_with(|g| g.bf16_tflops *= 2.0);
        let (s, t) = (g.step_cost(), g.tier_cost());
        assert_eq!(s.ns_per_pair, box_steps.ns_per_pair / 2.0);
        assert_eq!(s.step_ms, box_steps.step_ms);
        cheaper(&t);
        let quadratic = t.recompute.ns_per_token_squared / box_tier.recompute.ns_per_token_squared;
        assert!((0.45..0.6).contains(&quadratic), "{quadratic}");
        software(&s, &t);
        // Twice the link: half the transfer, the same fixed part.
        let g = the_box_with(|g| g.link.generation = 4);
        let (s, t) = (g.step_cost(), g.tier_cost());
        assert_eq!(
            t.cpu_restore.us_per_token,
            box_tier.cpu_restore.us_per_token / 2.0
        );
        assert_eq!((s, t.recompute), (box_steps.clone(), box_tier.recompute));
        // More memory: a larger pool, by what it adds. The L4's ECC leaves it 22.5 of its 24 GiB.
        let g = the_box_with(|g| g.memory_mib = 24 << 10);
        assert_eq!(
            g.pool_tokens(&QWEN),
            QWEN.blocks_in_bytes(13 << 30) * QWEN.block_tokens
        );
        let g = the_box_with(|g| g.memory_mib = L4.memory_mib);
        assert_eq!(g.pool_tokens(&QWEN), L4.pool_tokens(&QWEN));
        assert_eq!(
            g.pool_tokens(&QWEN),
            QWEN.blocks_in_bytes(23 << 29) * QWEN.block_tokens
        );
        // The L4 has less bandwidth than the box and more of everything else, but runs its linear
        // layers through Marlin at the BF16 rate, slower than the box's FP8; the H100 has more.
        let (l4, h100) = (L4.step_cost(), H100_SXM.step_cost());
        assert!(l4.step_ms > box_steps.step_ms && h100.step_ms < box_steps.step_ms);
        assert!(l4.us_per_token > box_steps.us_per_token);
        assert!(h100.us_per_token < box_steps.us_per_token);
        for s in [&l4, &h100] {
            assert!(s.ns_per_pair < box_steps.ns_per_pair);
        }
        let restore = |g: Gpu| g.tier_cost().cpu_restore.us_per_token;
        assert!(restore(H100_SXM) < restore(L4) && restore(L4) < restore(RTX_5060_TI));
        assert_eq!(
            (L4.pool_tokens(&QWEN), H100_SXM.pool_tokens(&QWEN)),
            (83_728, 502_432)
        );
    }

    #[test]
    fn the_l4_loses_a_sixteenth_of_its_memory_to_ecc() {
        // ECC, on by default, keeps a sixteenth of the L4's 24 GiB of GDDR6. The 22.5 GiB left, less
        // what the box keeps beside its pool, hold 5,233 blocks: 83,728 tokens, where all 24 GiB
        // would hold 94,656.
        let left = (24 << 30) / 16 * 15 - BOX_BESIDE_POOL;
        assert_eq!(QWEN.blocks_in_bytes(left), 5_233);
        assert_eq!(L4.pool_tokens(&QWEN), 83_728);
        let all = QWEN.blocks_in_bytes((24 << 30) - BOX_BESIDE_POOL) * QWEN.block_tokens;
        assert_eq!(all, 94_656);
        // The table gives the memory left; the box's and the H100's keep all theirs.
        let gpus = profiles(&GPUS);
        assert!(
            gpus.contains("| L4 24 GB | modeled | 22.5 GiB | 300 GB/s |"),
            "{gpus}"
        );
        assert!(
            gpus.contains("| H100 SXM 80 GB | modeled | 80 GiB | 3,350 GB/s |"),
            "{gpus}"
        );
    }

    #[test]
    fn the_modeled_cold_prefill_follows_the_scaled_step_loop() {
        // E0's host time, by its own derivation: 22.7 ms + 3.11 µs a token.
        let host = StepCost::frozen_box().e0_host(&TierCost::frozen_box());
        assert!((host.ms - 22.67).abs() < 0.01 && (host.us_per_token - 3.113).abs() < 0.001);
        // On the box, E0's fit runs 2% below its step loop's lone prefill at 4k and 8k tokens and
        // up to 14% above it at 1k; on every modeled card it keeps that relation to the scaled loop,
        // within the refit's 5%, at E0's lengths and beyond.
        let box_tier = TierCost::frozen_box();
        let relation = |steps: &StepCost, tier: &TierCost, tokens| {
            tier.recompute_ms(tokens) / lone_prefill_ms(steps, None, tier, tokens)
        };
        let on_box = |tokens| relation(&StepCost::frozen_box(), &box_tier, tokens);
        assert!((on_box(4_096) - 0.981).abs() < 0.005 && (on_box(1_024) - 1.138).abs() < 0.005);
        let cards = [
            (L4, QWEN3_4B_FP8, Scaling::MAP),
            (A100_40GB, QWEN3_4B_FP8, Scaling::MAP),
            (H100_SXM, QWEN3_4B_FP8, Scaling::MAP),
            (A100_40GB, QWEN3_4B_FP8, Scaling::central(&A100_40GB)),
            (A100_40GB, QWEN3_14B_FP8, Scaling::central(&A100_40GB)),
        ];
        for (gpu, model, scaling) in cards {
            let (steps, tier) = (
                gpu.step_cost_with(&model, scaling),
                gpu.tier_cost_with(&model, scaling),
            );
            for tokens in RECOMPUTE_LENGTHS {
                let (card, base) = (relation(&steps, &tier, tokens), on_box(tokens));
                assert!(
                    (card / base - 1.0).abs() < 0.05,
                    "{} {} at {tokens}: {card:.3} against the box's {base:.3}",
                    gpu.name,
                    model.name
                );
            }
        }
        // The E0 host time the loop runs with.
        assert_eq!(
            StepCost {
                host,
                ..StepCost::frozen_box()
            }
            .alone_ttft_ms(
                &box_tier,
                PromptSources {
                    computed: 4_096,
                    ..PromptSources::default()
                },
                512
            ),
            lone_prefill_ms(&StepCost::frozen_box(), None, &box_tier, 4_096)
        );
        // The restore on a Gen4 x16 link, against vLLM's on an L20 (34, 65 and 104 ms).
        let l4 = L4.tier_cost();
        for (tokens, measured) in [(4_096, 34.0), (8_192, 65.0), (16_384, 104.0)] {
            let modeled = l4.restore_ms(tokens);
            assert!(
                (modeled / measured - 1.0).abs() < 0.10,
                "{tokens}: {modeled:.1}"
            );
        }
    }

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
        let small = the_box_with(|g| g.memory_mib = 8 << 10);
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
    fn gpus_parse_by_name() {
        for g in CARDS {
            assert_eq!(g.name.parse::<Gpu>(), Ok(g));
        }
        // The map's cards are every card but the A100 80 GB.
        assert_eq!(
            CARDS
                .iter()
                .filter(|g| !GPUS.contains(g))
                .collect::<Vec<_>>(),
            [&A100_80GB]
        );
        let err = "a100".parse::<Gpu>().unwrap_err();
        assert!(
            err.contains("rtx-5060-ti, l4, a100-40gb, a100-80gb, h100-sxm"),
            "{err}"
        );
        for m in MODELS {
            assert_eq!(m.name.parse::<Model>(), Ok(m));
        }
        assert!("qwen3-8b".parse::<Model>().is_err());
    }

    #[test]
    fn the_box_checkpoint_runs_marlin_at_the_bf16_rate_on_sm80_and_sm89() {
        // PR #61224's four layers at M = 512 on an L4: 5,046.9 µs on Marlin, 3,563.7 in BF16.
        assert!((MARLIN_FACTOR - 1.416_197).abs() < 1e-6);
        // The box's attention term is 88% of its BF16 peak, its weight read 85% of its fixed step.
        assert!((BOX_ATTENTION_EFFICIENCY - 0.877_467).abs() < 1e-6);
        assert!((BOX_READ_SHARE - 0.851_535).abs() < 1e-6);
        let close = |x: f64, y: f64| (x / y - 1.0).abs() < 1e-5;
        // The A100 under the map's scaling: the box's terms by bandwidth (3.33 ms, 123.7 ns), by the
        // BF16 rate over Marlin's factor (41.3 µs: Marlin at 312 / 1.42 TFLOPS against the box's FP8
        // at 189.75) and by BF16 (2.15 ns), and the restore by the Gen4 x16 link (5.91 µs).
        let a = A100_40GB.step_cost();
        assert!(close(a.step_ms, 3.330_469_5) && close(a.us_per_token, 41.333_435_4));
        assert!(close(a.ns_per_pair, 2.154_453_1) && close(a.ns_per_context, 123.682_572_3));
        assert!((A100_40GB.tier_cost().cpu_restore.us_per_token - 5.91).abs() < 0.005);
        assert_eq!(A100_40GB.pool_tokens(&QWEN), 211_168);
        // The L4's linear layers, through Marlin, take 106.6 µs a token where FP8 would take 37.5.
        assert!(close(L4.step_cost().us_per_token, 106.578_775_6));
        // The forecast's range for the A100: Marlin as fast as BF16 (29.2 µs), the overhead kept
        // (4.55 ms), and FlashAttention 2 at 73% and 50% of the peak (2.59 and 3.78 ns).
        let low = A100_40GB.step_cost_with(&QWEN3_4B_FP8, Scaling::LOW);
        assert!(close(low.us_per_token, 29.186_226) && close(low.step_ms, 3.330_469_5));
        let central = A100_40GB.step_cost_with(&QWEN3_4B_FP8, Scaling::central(&A100_40GB));
        assert!(close(central.step_ms, 4.552_262_9) && close(central.ns_per_pair, 2.589_673_3));
        assert!(close(central.us_per_token, a.us_per_token));
        let high = A100_40GB.step_cost_with(&QWEN3_4B_FP8, Scaling::high(&A100_40GB));
        assert!(close(high.ns_per_pair, 3.780_923_1) && high.step_ms == central.step_ms);
        // A card with no published FlashAttention 2 figure keeps the box's efficiency.
        assert_eq!(Scaling::central(&L4).attention_efficiency, None);
        let l4 = L4.step_cost_with(&QWEN3_4B_FP8, Scaling::central(&L4));
        assert_eq!(l4.ns_per_pair, L4.step_cost().ns_per_pair);
    }

    #[test]
    fn the_a100_80gb_is_the_40gb_with_its_memory_and_bandwidth() {
        // Every field but the name, the memory and the bandwidth is the 40 GB card's.
        assert_eq!(
            Gpu {
                name: A100_40GB.name,
                title: A100_40GB.title,
                memory_mib: A100_40GB.memory_mib,
                bandwidth_gb_s: A100_40GB.bandwidth_gb_s,
                ..A100_80GB
            },
            A100_40GB
        );
        assert_eq!(
            (A100_80GB.memory_mib, A100_80GB.bandwidth_gb_s),
            (81_920, 2_039.0)
        );
        // The bandwidth terms shrink by 1,555 / 2,039: the fixed step from 3.33 to 2.54 ms, a KV
        // token read from 123.7 to 94.3 ns. The linear layers, attention and the restore stay.
        let close = |x: f64, y: f64| (x / y - 1.0).abs() < 1e-12;
        let ratio = A100_40GB.bandwidth_gb_s / A100_80GB.bandwidth_gb_s;
        for s in [Scaling::MAP, Scaling::LOW] {
            let (a40, a80) = (
                A100_40GB.step_cost_with(&QWEN3_4B_FP8, s),
                A100_80GB.step_cost_with(&QWEN3_4B_FP8, s),
            );
            assert!(close(a80.step_ms, a40.step_ms * ratio), "{}", a80.step_ms);
            assert!(close(a80.ns_per_context, a40.ns_per_context * ratio));
            assert_eq!(
                (a80.us_per_token, a80.ns_per_pair),
                (a40.us_per_token, a40.ns_per_pair)
            );
        }
        let a80 = A100_80GB.step_cost();
        assert!((a80.step_ms - 2.5399).abs() < 1e-4 && (a80.ns_per_context - 94.324).abs() < 1e-3);
        assert_eq!(
            A100_80GB.tier_cost().cpu_restore,
            A100_40GB.tier_cost().cpu_restore
        );
        // The forecast's central setting keeps the box's 1.7 ms of overhead beside the faster read:
        // 3.88 ms against the 40 GB card's 4.55; FlashAttention 2's range is the A100's.
        let central = A100_80GB.step_cost_with(&QWEN3_4B_FP8, Scaling::central(&A100_80GB));
        assert!(
            (central.step_ms - 3.879_07).abs() < 1e-4,
            "{}",
            central.step_ms
        );
        assert_eq!(Scaling::high(&A100_80GB), Scaling::high(&A100_40GB));
        // 80 GiB, the H100's: 31,402 blocks of the 4B (502,432 tokens) and 23,718 of the 14B
        // (379,488), and the 32x corpus replays at 4x, as on the H100.
        assert_eq!(A100_80GB.pool_tokens(&QWEN), 502_432);
        assert_eq!(A100_80GB.pool_tokens(&QWEN), H100_SXM.pool_tokens(&QWEN));
        assert_eq!(A100_80GB.model_pool_tokens(&QWEN3_14B_FP8), 379_488);
        assert_eq!(fitted_token_scale(&A100_80GB, &QWEN, 31_144, 32), Some(8));
        // Its row of the profiles' table.
        let row = profiles(&[A100_80GB]);
        assert!(
            row.contains(
                "| A100 SXM4 80 GB | modeled | 80 GiB | 2,039 GB/s | none / 312.0 TFLOPS \
                 | Marlin W8A16, 1.42× a BF16 GEMM | PCIe 4.0 x16, 31.5 GB/s | 502,432 tokens \
                 | 2.54 ms | 41.33 µs | 2.15 ns | 94.3 ns | 10.3 ms + 5.91 µs/token |"
            ),
            "{row}"
        );
    }

    /// The box's host as bench_host.py measured it on Phastform.
    fn box_host() -> Host {
        Host::from_bench_host(include_str!("../testdata/box-host.json")).unwrap()
    }

    #[test]
    fn a_hosts_file_gives_its_cpu_and_one_copy_rate_and_is_refused_without_either() {
        // The box's own: its CPU, and the median and the one-copy rate the box's constants hold, so
        // that the box's factors are 1.
        let b = box_host();
        assert_eq!(
            b.cpu.as_deref(),
            Some("Intel(R) Xeon(R) CPU E5-2609 v4 @ 1.70GHz")
        );
        assert_eq!(
            (b.python_ms, b.h2d_gb_per_s),
            (BOX_PYTHON_MS, BOX_H2D_GB_PER_S)
        );
        assert_eq!((b.cpu_ratio(), b.link_ratio()), (1.0, 1.0));
        // Of the link's rows, the one host to device in one copy; nothing else is read.
        let paged = json!({"direction": "h2d", "page_bytes": 32_768, "gb_per_s": 10.5});
        let d2h = json!({"direction": "d2h", "page_bytes": null, "gb_per_s": 12.9});
        let one_copy = json!({"direction": "h2d", "page_bytes": null, "gb_per_s": 12.1});
        let python = json!({"p50": 190.0, "min": 188.0, "reps": 7});
        let file = json!({
            "python_ms": python,
            "link": [paged, d2h, one_copy],
            "gpu": {"name": "NVIDIA A100-SXM4-80GB"},
        });
        let host = Host::from_bench_host(&file.to_string()).unwrap();
        assert_eq!(
            host,
            Host {
                cpu: None,
                python_ms: 190.0,
                h2d_gb_per_s: 12.1
            }
        );
        // Refused, naming what is missing: the median, or the row (a file bench_host.py wrote with
        // --no-gpu has no link; one whose link failed has no rows; paged and d2h rows are not it).
        let refused =
            |file: serde_json::Value| Host::from_bench_host(&file.to_string()).unwrap_err();
        for file in [
            json!({"link": [one_copy]}),
            json!({"python_ms": {"min": 188.0}, "link": [one_copy]}),
        ] {
            assert_eq!(refused(file), "python_ms.p50 is missing");
        }
        for file in [
            json!({"cpu": "Intel(R) Xeon(R) CPU @ 2.20GHz", "python_ms": python}),
            json!({"python_ms": python, "link": [], "link_error": "RuntimeError: no CUDA GPUs"}),
            json!({"python_ms": python, "link": [paged, d2h]}),
        ] {
            assert_eq!(
                refused(file),
                "the link's h2d row of one copy (page_bytes null) is missing"
            );
        }
        let no_rate = json!({"direction": "h2d", "page_bytes": null, "gb_per_s": null});
        assert_eq!(
            refused(json!({"python_ms": python, "link": [no_rate]})),
            "the link's h2d row of one copy has no gb_per_s"
        );
        assert_eq!(
            refused(json!({"python_ms": {"p50": 0.0}, "link": [one_copy]})),
            "python_ms.p50 is 0, not a positive number"
        );
        assert!(Host::from_bench_host("python_ms: 190").is_err());
    }

    #[test]
    fn on_the_boxs_own_host_a_card_keeps_its_costs_and_takes_the_boxs_link() {
        let b = box_host();
        let box_restore = TierCost::frozen_box().cpu_restore;
        for gpu in CARDS {
            for model in MODELS {
                for s in [
                    Scaling::MAP,
                    Scaling::LOW,
                    Scaling::central(&gpu),
                    Scaling::high(&gpu),
                ] {
                    // The box's CPU: every step term and the host time as without a host.
                    assert_eq!(
                        gpu.step_cost_on(&model, s, Some(&b)),
                        gpu.step_cost_with(&model, s)
                    );
                    // The box's measured link in place of the card's datasheet one: the box's
                    // restore, by the model's KV, and the rest as without a host.
                    let (on, without) = (
                        gpu.tier_cost_on(&model, s, Some(&b)),
                        gpu.tier_cost_with(&model, s),
                    );
                    assert_eq!(
                        on.cpu_restore,
                        Linear {
                            ms: box_restore.ms,
                            us_per_token: box_restore.us_per_token * model.over_box().kv_bytes,
                        }
                    );
                    assert_eq!(
                        TierCost {
                            cpu_restore: without.cpu_restore,
                            ..on
                        },
                        without
                    );
                }
            }
        }
        // So the box gets its measured costs back, E0's host time is the box's, and the A100
        // restores at the box's 23.64 µs a token where its Gen4 x16 link gives 5.91.
        assert_eq!(
            RTX_5060_TI.step_cost_on(&QWEN3_4B_FP8, Scaling::MAP, Some(&b)),
            StepCost::frozen_box()
        );
        assert_eq!(
            RTX_5060_TI.tier_cost_on(&QWEN3_4B_FP8, Scaling::MAP, Some(&b)),
            TierCost::frozen_box()
        );
        assert_eq!(e0_host(Some(&b)), e0_host(None));
        let central = Scaling::central(&A100_80GB);
        let a100 = |host| A100_80GB.tier_cost_on(&QWEN3_4B_FP8, central, host);
        assert_eq!(a100(Some(&b)).cpu_restore.us_per_token, 23.64);
        assert!((a100(None).cpu_restore.us_per_token - 5.91).abs() < 0.005);
    }

    #[test]
    fn a_hosts_cpu_scales_only_the_api_servers_time_and_its_link_only_the_transfer() {
        let (box_steps, box_tier) = (StepCost::frozen_box(), TierCost::frozen_box());
        let host = |python_ms, h2d_gb_per_s| Host {
            cpu: None,
            python_ms,
            h2d_gb_per_s,
        };
        let half = |l: Linear| Linear {
            ms: l.ms / 2.0,
            us_per_token: l.us_per_token / 2.0,
        };
        // Half the box's Python time: half the API server's time, on a prompt and beside each
        // decode, and half E0's; the step terms, the mixed-step extra, the stall and the restore stay.
        let fast = host(BOX_PYTHON_MS / 2.0, BOX_H2D_GB_PER_S);
        assert_eq!(fast.cpu_ratio(), 2.0);
        let steps = RTX_5060_TI.step_cost_on(&QWEN3_4B_FP8, Scaling::MAP, Some(&fast));
        assert_eq!(steps.host, half(box_steps.host));
        assert_eq!(steps.host_per_decode_ms, box_steps.host_per_decode_ms / 2.0);
        assert_eq!(
            StepCost {
                host: box_steps.host,
                host_per_decode_ms: box_steps.host_per_decode_ms,
                ..steps
            },
            box_steps
        );
        assert_eq!(e0_host(Some(&fast)), half(e0_host(None)));
        // The cold prefill is cheaper by E0's host time saved, most at the shortest prompts.
        let tier = RTX_5060_TI.tier_cost_on(&QWEN3_4B_FP8, Scaling::MAP, Some(&fast));
        assert_eq!(tier.cpu_restore, box_tier.cpu_restore);
        let saved = |tokens| 1.0 - tier.recompute_ms(tokens) / box_tier.recompute_ms(tokens);
        for tokens in RECOMPUTE_LENGTHS {
            assert!(saved(tokens) > 0.0, "{tokens}: {}", saved(tokens));
        }
        assert!(
            saved(1_024) > 0.05 && saved(31_744) < 0.01,
            "{}",
            saved(1_024)
        );
        // Twice the box's one-copy rate: half the restore's transfer, the same fixed part, and
        // nothing else.
        let wide = host(BOX_PYTHON_MS, 2.0 * BOX_H2D_GB_PER_S);
        assert_eq!(
            RTX_5060_TI.step_cost_on(&QWEN3_4B_FP8, Scaling::MAP, Some(&wide)),
            box_steps
        );
        let tier = RTX_5060_TI.tier_cost_on(&QWEN3_4B_FP8, Scaling::MAP, Some(&wide));
        assert_eq!(
            tier,
            TierCost {
                cpu_restore: Linear {
                    ms: box_tier.cpu_restore.ms,
                    us_per_token: box_tier.cpu_restore.us_per_token / 2.0,
                },
                ..box_tier.clone()
            }
        );
        // On every card the transfer is a token's KV at the host's rate times the share of the box's
        // own the box's E0 restore reaches: 92%, 6.24 of 6.76 GB/s.
        assert!(
            (BOX_RESTORE_SHARE - 0.9223).abs() < 1e-4,
            "{BOX_RESTORE_SHARE}"
        );
        for gpu in [A100_80GB, H100_SXM] {
            for model in MODELS {
                let t = gpu.tier_cost_on(&model, Scaling::central(&gpu), Some(&wide));
                let bytes = model.geometry.bytes_per_token as f64;
                let at_share = bytes / (wide.h2d_gb_per_s * 1e3 * BOX_RESTORE_SHARE);
                assert!((t.cpu_restore.us_per_token / at_share - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_model_scales_the_terms_by_its_bytes_and_flops() {
        let close = |x: f64, y: f64| (x / y - 1.0).abs() < 1e-5;
        // Qwen3-14B-FP8 on the box: 14.77 GB a step, 3.6 times the FLOPs a token, 1.39 times the
        // attention a pair and 160 KiB of KV a token.
        let b = RTX_5060_TI.step_cost_with(&QWEN3_14B_FP8, Scaling::MAP);
        assert!(close(b.step_ms, 34.676_071_4) && close(b.us_per_token, 174.401_072_9));
        assert!(close(b.ns_per_pair, 19.680_555_6) && close(b.ns_per_context, 477.0));
        let t = RTX_5060_TI.tier_cost_with(&QWEN3_14B_FP8, Scaling::MAP);
        assert!(close(
            t.cpu_restore.us_per_token,
            23.64 * 163_840.0 / 147_456.0
        ));
        assert_eq!(t.cpu_restore.ms, TierCost::frozen_box().cpu_restore.ms);
        // Its pool on an A100 40 GB: 11.9 GB more of weights beside it leave 7,334 blocks.
        assert_eq!(A100_40GB.model_pool_tokens(&QWEN3_14B_FP8), 117_344);
        assert_eq!(QWEN3_14B_FP8.geometry.bytes_per_token, 163_840);
        // No card with the box's footprint and the 14B's weights has room on 16 GiB.
        assert_eq!(RTX_5060_TI.model_pool_tokens(&QWEN3_14B_FP8), 0);
        // The 4B's pool is every card's own pool.
        for g in CARDS {
            assert_eq!(g.model_pool_tokens(&QWEN3_4B_FP8), g.pool_tokens(&QWEN));
        }
        // A 31,744-token cold prompt of the 14B on the A100 (central) takes 2.4 times the 4B's: 3.6
        // times the linear FLOPs, 1.4 times the attention, which dominates prompts this long.
        let central = Scaling::central(&A100_40GB);
        let cold = |model: &Model| {
            A100_40GB
                .tier_cost_with(model, central)
                .recompute_ms(31_744)
        };
        let ratio = cold(&QWEN3_14B_FP8) / cold(&QWEN3_4B_FP8);
        assert!((2.2..2.6).contains(&ratio), "{ratio}");
    }

    #[test]
    fn a_quadratic_refits_its_own_points() {
        let q = Quadratic {
            ms: 47.8,
            us_per_token: 63.11,
            ns_per_token_squared: 8.1258,
        };
        let points = RECOMPUTE_LENGTHS.map(|tokens| (tokens, q.at(tokens)));
        let fit = fit_quadratic(&points);
        assert!((fit.ms / q.ms - 1.0).abs() < 1e-6, "{fit:?}");
        assert!(
            (fit.us_per_token / q.us_per_token - 1.0).abs() < 1e-6,
            "{fit:?}"
        );
        assert!(
            (fit.ns_per_token_squared / q.ns_per_token_squared - 1.0).abs() < 1e-6,
            "{fit:?}"
        );
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
