//! The live working set: the distinct full blocks in the latest prompt of every live stream. Over the profiling
//! window the replay integrates its size, keeps its peak, and measures how long it, and the blocks running
//! requests pin, leave less than a realtime tier's KV of the pool free.

use super::ReplayStats;
use crate::engine::Engine;
use crate::prefix::{BlockId, IdMap};

#[derive(Debug)]
pub(super) struct WorkingSet {
    /// Each block with the number of live streams whose latest prompt holds it.
    blocks: IdMap<u32>,
    /// Where `integral` has reached.
    last_change: u64,
    /// The working set's size integrated over the profiling window, in blocks × ns.
    integral: f64,
    /// Its peak size in the profiling window, in blocks.
    peak: usize,
    /// [`ReplayConfig::realtime_tokens`](super::ReplayConfig::realtime_tokens) in blocks, and the profiling time
    /// (ns) in which the working set, and the pinned blocks, left fewer than that free.
    realtime_blocks: usize,
    realtime_short_ns: u64,
    realtime_blocked_ns: u64,
}

impl WorkingSet {
    /// An empty working set, measuring the time it leaves fewer than `realtime_blocks` free (none for 0).
    pub fn new(realtime_blocks: usize) -> Self {
        Self {
            blocks: IdMap::default(),
            last_change: 0,
            integral: 0.0,
            peak: 0,
            realtime_blocks,
            realtime_short_ns: 0,
            realtime_blocked_ns: 0,
        }
    }

    /// A stream's latest prompt is `prompt`, which joins the set.
    pub fn add(&mut self, prompt: &[BlockId]) {
        for &b in prompt {
            *self.blocks.entry(b).or_insert(0) += 1;
        }
    }

    /// A stream's former latest prompt, which leaves the set.
    pub fn remove(&mut self, prompt: &[BlockId]) {
        for b in prompt {
            if let Some(n) = self.blocks.get_mut(b) {
                *n -= 1;
                if *n == 0 {
                    self.blocks.remove(b);
                }
            }
        }
    }

    /// Profiling starts at `at`: the integral starts there, and the peak from the set's size then.
    pub fn start(&mut self, at: u64) {
        self.last_change = at;
        self.peak = self.blocks.len();
    }

    /// Notes the set's size for its peak, in the profiling window.
    pub fn observe_peak(&mut self) {
        self.peak = self.peak.max(self.blocks.len());
    }

    /// Integrates the set, and the time it and `engine`'s pinned blocks leave the realtime tier short, from where
    /// the integral has reached to `at`, within the profiling window `[start, end]`.
    pub fn advance(&mut self, at: u64, start: u64, end: u64, engine: &Engine) {
        if at <= self.last_change {
            return;
        }
        let from = self.last_change.max(start);
        let to = at.min(end);
        if to > from {
            self.integral += self.blocks.len() as f64 * (to - from) as f64;
            if self.realtime_blocks > 0 {
                let pool = engine.gpu_capacity();
                if self.blocks.len() + self.realtime_blocks > pool {
                    self.realtime_short_ns += to - from;
                }
                if engine.pinned_blocks() + self.realtime_blocks > pool {
                    self.realtime_blocked_ns += to - from;
                }
            }
        }
        self.last_change = at;
    }

    /// Writes the set's mean and peak over a profiling window of `window` ns, and the realtime tier's shares of it,
    /// into `stats`, in tokens of `block_tokens`.
    pub fn finish(&self, window: f64, block_tokens: u64, stats: &mut ReplayStats) {
        stats.working_set_mean_tokens = self.integral / window * block_tokens as f64;
        stats.working_set_peak_tokens = self.peak as u64 * block_tokens;
        if self.realtime_blocks > 0 {
            stats.realtime_short_share = Some(self.realtime_short_ns as f64 / window);
            stats.realtime_blocked_share = Some(self.realtime_blocked_ns as f64 / window);
        }
    }
}
