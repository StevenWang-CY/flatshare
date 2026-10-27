//! AIPerf 0.13.0's `agentic_replay` timing (`strategies/agentic_replay.py`, `branch_orchestrator.py`,
//! `replay_dependencies.py`, `session_tree.py`), run over the engine model as a discrete-event
//! simulation.
//!
//! - **Lanes.** `concurrency` lanes take traces from a sequential sampler over the corpus (AIPerf's
//!   Weka loader samples sequentially); each lane snapshots its trace at its own `t*`.
//! - **Warmup.** Every stream live at `t*` first sends its last request before `t*` with
//!   `max_tokens = 1`, all at once (burst starts) or aligned on `t*` (spread). Profiling starts when the
//!   last warmup request returns, and the duration counts from there.
//! - **Dispatch.** Each live stream sends its next turn at its recorded offset from `t*`; with burst
//!   starts the lane's earliest one goes at once. A turn's successor goes its recorded delay after it
//!   returns. A root turn spawns its branches' children when it returns, or when it is sent if the
//!   recorded spawn overlapped the request; each child starts at its recorded offset from the spawn. A
//!   root turn gated on children waits until they finish and then goes at once.
//! - **Barriers.** A turn whose replay predecessors (in its scope) have not completed is held until
//!   they have.
//! - **Recycling.** When a lane's session tree (root and every child) has drained, the lane starts the
//!   next trace from turn 0 under a fresh namespace (AIPerf's cache-bust marker).
//! - **Stop.** No request is sent after the duration; requests still running at the end of the grace
//!   period are dropped from the results. When no request is in flight, pending scheduler timers are
//!   pulled forward so the engine idles at most `system_idle_cap` seconds.
//!
//! The engine serves requests in one of two ways ([`Service`]). Under contention, a request reaches
//! vLLM's step loop ([`crate::scheduler`]) after its host time and shares every step with the
//! requests running beside it. Idle (v0), its time to first token is E0's cost for its hit pattern
//! after any wait for free blocks, then it decodes at a constant inter-token latency, with no
//! queueing behind other requests.
//!
//! A session pauses between turns for the trace's delay, or for a TraceLab tool wait in its place
//! ([`Pauses`]); a [`Retention`] arm acts on its KV meanwhile ([`crate::retention`]).
//!
//! This module holds the event loop and AIPerf's dispatch. `clock` holds the pending events and how far the idle
//! cap has pulled them forward, `tree` the session trees with their gates and barriers, `retain` what a retention
//! arm does with a pausing session's KV, and `working_set` the live working set and the time it leaves a realtime
//! tier short of KV.

mod clock;
mod retain;
#[cfg(test)]
mod tests;
mod tree;
mod working_set;

use std::rc::Rc;

use flatshare_core::{PromptSources, StepCost, TierCost};
use serde::Serialize;

use self::clock::Queue;
use self::retain::RetainStep;
use self::tree::{Kind, Lane, Play, Stream};
use self::working_set::WorkingSet;
use crate::aiperf::{ConvKind, ReplayPlan, snapshot};
use crate::corpus::Corpus;
use crate::engine::{Admission, Engine, EngineRequest};
use crate::pauses::ToolPauses;
use crate::prefix::{BlockId, IdMap, IdSet, MAX_PLAYS, NodeId, Tokens};
use crate::retention::{PauseHistory, Retention};
use crate::scheduler::{Outcome, Scheduler, StepBudget};

/// How the engine serves the requests it admits.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Service {
    /// v0: E0's idle-box cost per request ([`TierCost`]), with no queueing behind other requests.
    Idle,
    /// vLLM's step loop with the step-time model.
    Contention { budget: StepBudget, steps: StepCost },
}

impl Service {
    /// The step loop on the frozen box: its 512/256 step budget and the fitted step-time model.
    pub const fn frozen_box() -> Self {
        Self::Contention {
            budget: StepBudget::FROZEN_BOX,
            steps: StepCost::frozen_box(),
        }
    }
}

/// Where the pauses between a session's turns come from.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pauses {
    /// The traces' delays (AIPerf's think times).
    Trace,
    /// A TraceLab tool wait ([`ToolPauses`]) in place of every delay, capped at `cap_s`.
    TraceLab { cap_s: Option<f64> },
}

/// How the replay is driven: the `phases` section of `configs/agentic_replay.yaml`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReplayConfig {
    /// Live session trees.
    pub concurrency: usize,
    pub duration_s: f64,
    pub grace_s: f64,
    pub start_min_ratio: f64,
    pub start_max_ratio: f64,
    /// `burstPhaseStarts`.
    pub burst_starts: bool,
    /// `systemIdleGapCapSeconds`.
    pub system_idle_cap_s: Option<f64>,
    /// `randomSeed`.
    pub seed: u64,
    /// Recycle no lane once this many session plays have started, the initial lanes included; every
    /// lane starts its first play. `None` for no limit.
    pub max_sessions: Option<u64>,
    /// Tokens per engine block.
    pub block_tokens: u32,
    /// Multiplies the tokens of every hash block and every output: 32 replays a 32x-scaled corpus at
    /// its original token counts (output lengths come back rounded up to a multiple of 32).
    pub token_scale: u32,
    pub pauses: Pauses,
    /// KV a realtime tier needs free beside the replay (E2's voice calls), in tokens: the replay
    /// measures how long the background leaves less than this of the pool free. 0 measures nothing.
    #[serde(skip_serializing_if = "is_zero")]
    pub realtime_tokens: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

impl Default for ReplayConfig {
    /// The W0 run: 16 session trees, 600 s plus 60 s grace, burst starts, seed 11.
    fn default() -> Self {
        Self {
            concurrency: 16,
            duration_s: 600.0,
            grace_s: 60.0,
            start_min_ratio: 0.0,
            start_max_ratio: 1.0,
            burst_starts: true,
            system_idle_cap_s: Some(10.0),
            seed: 11,
            max_sessions: None,
            block_tokens: 16,
            token_scale: 1,
            pauses: Pauses::Trace,
            realtime_tokens: 0,
        }
    }
}

/// One completed request.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestRecord {
    pub trace: u32,
    pub conv: u32,
    pub turn: u32,
    pub kind: ConvKind,
    pub warmup: bool,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub sources: PromptSources,
    /// Computed tokens of full blocks this run had already seen (evicted from every tier, or the
    /// recomputed last block of a fully cached prompt).
    pub recomputed: u64,
    /// The think-time delay that preceded this turn, in ms.
    pub delay_ms: Option<f64>,
    pub issued_ns: u64,
    /// When the engine took its blocks.
    pub admitted_ns: u64,
    /// How long it waited for free blocks.
    pub blocked_ns: u64,
    pub done_ns: u64,
    /// Time to first token as the client sees it.
    pub ttft_ns: u64,
}

/// The engine-side order of a run, for replaying it under another policy.
#[derive(Clone, Debug, Default)]
pub struct Timeline {
    /// The admitted requests, in admission order.
    pub requests: Vec<TimelineRequest>,
    /// Admissions and finishes, in the order the engine saw them.
    pub events: Vec<TimelineEvent>,
}

/// One admitted request of a [`Timeline`].
#[derive(Clone, Debug)]
pub struct TimelineRequest {
    pub prompt: Vec<BlockId>,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    /// Index into the run's records; `None` for warmup requests and requests that did not complete
    /// in time.
    pub record: Option<usize>,
    pub warmup: bool,
    /// Where the prompt comes from: the corpus trace, its conversation and turn, and the KV
    /// namespace (the session play) its blocks were cut in.
    pub trace: u32,
    pub conv: u32,
    pub turn: u32,
    pub play: u64,
}

/// An admission or a finish of a [`Timeline`] request, by its index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineEvent {
    Admit(usize),
    Finish(usize),
}

/// How the replay itself went: lanes and plays, warmup, barriers, the idle cap, the working set.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReplayStats {
    pub lanes: usize,
    pub sessions_started: u64,
    pub warmup_requests: u64,
    /// When profiling started, in seconds from the run's start: how long the warmup took.
    pub profiling_start_s: f64,
    /// Requests sent during profiling but still running when the grace period ended.
    pub cut_off_requests: u64,
    /// Turns held by replay barriers at some point.
    pub barrier_holds: u64,
    /// The idle cap's jumps, and the idle time they skipped, until sending closes.
    pub idle_cap_jumps: u64,
    pub idle_cap_skipped_s: f64,
    /// Time-weighted mean and peak of the tokens in the latest prompt of every live stream (distinct
    /// full blocks), over the profiling window.
    pub working_set_mean_tokens: f64,
    pub working_set_peak_tokens: u64,
    /// The share of the profiling window in which the live working set left less than
    /// [`ReplayConfig::realtime_tokens`] of the pool free: the realtime tier's KV then displaces
    /// cached background KV, which vLLM evicts for a request that needs blocks. `None`, and left out,
    /// when nothing is measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub realtime_short_share: Option<f64>,
    /// The share in which the blocks running requests pin left less than that free: a realtime
    /// request then waits for blocks, as vLLM preempts no running request for a waiting one (#40004).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub realtime_blocked_share: Option<f64>,
}

/// What one replay produced.
#[derive(Debug)]
pub struct ReplayOutput {
    pub records: Vec<RequestRecord>,
    pub stats: ReplayStats,
    pub engine: crate::engine::EngineStats,
    pub timeline: Option<Timeline>,
}

#[derive(Clone, Copy, Debug)]
enum Event {
    Dispatch {
        stream: u32,
        turn: u32,
        kind: Kind,
        delay_ms: Option<f64>,
    },
    Warm {
        stream: u32,
        turn: u32,
    },
    /// Idle service: a request's last token.
    Complete(u64),
    /// Contention: a request reaches the scheduler after its host time.
    Arrive(u64),
    /// Contention: the step loop runs (a step ends, or a restore finishes on an idle engine).
    Step,
    Recycle(usize),
    /// A retention timer of pause `pause` of `stream`.
    Retain {
        stream: u32,
        pause: u64,
        step: RetainStep,
    },
}

/// A request on its way to the engine, in it, or waiting for its blocks.
#[derive(Debug)]
struct InFlight {
    stream: u32,
    turn: u32,
    warmup: bool,
    issued: u64,
    admitted: u64,
    blocked: u64,
    /// When its first token came out, under contention.
    first_token: u64,
    prompt: Vec<BlockId>,
    prompt_tokens: u64,
    output_tokens: u64,
    delay_ms: Option<f64>,
    admission: Option<Admission>,
    recomputed: u64,
    /// Its index in the timeline, once admitted into a run that records one.
    timeline: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Warmup,
    Profiling,
}

/// Computed tokens of an admitted prompt in full blocks that `seen` holds: blocks some earlier
/// request had, now evicted from every tier (or the always-recomputed last block). Adds the prompt's
/// blocks to `seen`.
pub(crate) fn recomputed_tokens(
    seen: &mut IdSet,
    prompt: &[BlockId],
    adm: Admission,
    block_tokens: u64,
) -> u64 {
    let start = (adm.gpu_blocks + adm.cpu_blocks) as usize;
    let again = prompt[start..].iter().filter(|b| seen.contains(b)).count() as u64;
    seen.extend(prompt.iter().copied());
    again * block_tokens
}

fn ns(ms: f64) -> u64 {
    if ms.is_finite() && ms > 0.0 {
        (ms * 1e6).round() as u64
    } else {
        0
    }
}

/// Runs one replay of `corpus` through `engine`.
#[derive(Debug)]
pub struct Replay<'a> {
    cfg: ReplayConfig,
    corpus: &'a mut Corpus,
    engine: Engine,
    cost: TierCost,
    /// The step loop under contention; `None` for the idle service.
    scheduler: Option<Scheduler>,
    /// When the step loop runs next: a `Step` event due at another time is stale.
    step_at: Option<u64>,
    /// A reused buffer for the step loop's outcomes.
    outcomes: Vec<Outcome>,
    retention: Retention,
    /// The share of a restore for which the step loop stalls: the step model's, 1 when idle.
    restore_stall: f64,
    /// Every session's past pauses: the history information level.
    history: PauseHistory,
    next_pause: u64,
    now: u64,
    /// The pending events.
    queue: Queue<Event>,
    phase: Phase,
    profiling_start: u64,
    profiling_end: u64,
    grace_end: u64,
    plays: Vec<Play>,
    streams: Vec<Stream>,
    lanes: Vec<Lane>,
    /// Draws so far from the sequential trace sampler.
    sampler: usize,
    in_flight: IdMap<InFlight>,
    warmup_outstanding: usize,
    /// The start of the current idle gap (nothing in flight since the last return), for the idle cap.
    idle_since: Option<u64>,
    next_request: u64,
    records: Vec<RequestRecord>,
    timeline: Option<Timeline>,
    /// Every block an admitted prompt has had, for attributing recompute.
    seen: IdSet,
    /// The blocks of the live streams' latest prompts.
    working_set: WorkingSet,
    stats: ReplayStats,
    /// A reused buffer for prefix-tree paths.
    scratch: Vec<NodeId>,
}

impl<'a> Replay<'a> {
    /// A replay of `corpus` through `engine` with `retention` acting on pausing sessions; with
    /// `record`, the output keeps the engine's timeline.
    pub fn new(
        cfg: ReplayConfig,
        corpus: &'a mut Corpus,
        engine: Engine,
        cost: TierCost,
        service: Service,
        retention: Retention,
        record: bool,
    ) -> Self {
        let (scheduler, restore_stall) = match service {
            Service::Idle => (None, 1.0),
            Service::Contention { budget, steps } => {
                let stall = steps.restore_stall;
                (Some(Scheduler::new(budget, steps, cost.clone())), stall)
            }
        };
        let realtime_blocks = cfg.realtime_tokens.div_ceil(cfg.block_tokens as u64) as usize;
        Self {
            cfg,
            corpus,
            engine,
            cost,
            scheduler,
            step_at: None,
            outcomes: Vec::new(),
            retention,
            restore_stall,
            history: PauseHistory::default(),
            next_pause: 0,
            now: 0,
            queue: Queue::default(),
            phase: Phase::Warmup,
            profiling_start: 0,
            profiling_end: u64::MAX,
            grace_end: u64::MAX,
            plays: Vec::new(),
            streams: Vec::new(),
            lanes: Vec::new(),
            sampler: 0,
            in_flight: IdMap::default(),
            warmup_outstanding: 0,
            idle_since: None,
            next_request: 0,
            records: Vec::new(),
            timeline: record.then(Timeline::default),
            seen: IdSet::default(),
            working_set: WorkingSet::new(realtime_blocks),
            stats: ReplayStats::default(),
            scratch: Vec::new(),
        }
    }

    /// Runs the replay to the end of its grace period, or until it runs out of work.
    pub fn run(mut self) -> Result<ReplayOutput, String> {
        self.start_lanes()?;
        while let Some((at, event, from_timers)) = self.queue.pop(self.now) {
            if self.phase == Phase::Profiling && at > self.grace_end {
                break;
            }
            self.advance(at);
            match event {
                Event::Dispatch {
                    stream,
                    turn,
                    kind,
                    delay_ms,
                } => self.submit(stream, turn, kind, delay_ms),
                Event::Warm { stream, turn } => self.issue(stream, turn, true, None),
                Event::Complete(id) => self.complete(id)?,
                Event::Arrive(id) => self.arrive(id)?,
                Event::Step if self.step_at == Some(at) => self.run_steps()?,
                Event::Step => {}
                Event::Recycle(lane) => self.recycle(lane)?,
                Event::Retain {
                    stream,
                    pause,
                    step,
                } => self.retain(stream, pause, step)?,
            }
            if from_timers {
                self.enforce_idle_cap(false);
            }
        }
        if self.engine.stats.oversized > 0 {
            return Err(format!(
                "{} requests need more KV blocks than the GPU pool's {}",
                self.engine.stats.oversized,
                self.engine.gpu_capacity()
            ));
        }
        // The working-set mean covers the duration, or the part of it before the run ran out of work;
        // `advance` has integrated up to the last event.
        let window_end = self.now.min(self.profiling_end);
        self.stats.cut_off_requests = self
            .in_flight
            .values()
            .filter(|f| !f.warmup && f.issued >= self.profiling_start)
            .count() as u64;
        let window = window_end.saturating_sub(self.profiling_start).max(1) as f64;
        self.working_set
            .finish(window, self.cfg.block_tokens as u64, &mut self.stats);
        self.stats.lanes = self.lanes.len();
        Ok(ReplayOutput {
            records: self.records,
            stats: self.stats,
            engine: self.engine.stats,
            timeline: self.timeline,
        })
    }

    /// Moves the clock to `at`, integrating the live working set, and the time it and the pinned
    /// blocks leave the realtime tier short, over the profiling duration.
    fn advance(&mut self, at: u64) {
        if self.phase == Phase::Profiling {
            self.working_set
                .advance(at, self.profiling_start, self.profiling_end, &self.engine);
        }
        self.now = self.now.max(at);
    }

    fn sending_open(&self) -> bool {
        self.phase == Phase::Profiling && self.now < self.profiling_end
    }

    fn can_start_session(&self) -> bool {
        self.sending_open()
            && self
                .cfg
                .max_sessions
                .is_none_or(|m| self.stats.sessions_started < m)
    }

    /// The next spawnable trace from the sequential sampler.
    fn next_trace(&mut self) -> Result<(usize, Rc<ReplayPlan>), String> {
        for _ in 0..self.corpus.len() {
            let idx = self.sampler % self.corpus.len();
            self.sampler += 1;
            let plan = self.corpus.plan(idx)?;
            if !plan.root().turns.is_empty() {
                return Ok((idx, plan));
            }
        }
        Err("no trace has a main-agent request".into())
    }

    fn new_play(&mut self, trace: usize, plan: Rc<ReplayPlan>, lane: usize) -> Result<u32, String> {
        let namespace = self.plays.len() as u64;
        if namespace >= MAX_PLAYS {
            return Err("too many session plays for the block namespace".into());
        }
        self.plays
            .push(Play::new(plan, trace as u32, lane, namespace));
        self.stats.sessions_started += 1;
        Ok(namespace as u32)
    }

    fn new_stream(&mut self, play: u32, conv: u32, depth: u8) -> u32 {
        self.streams.push(Stream {
            play,
            conv,
            depth,
            joins: Vec::new(),
            done: false,
            context: Vec::new(),
            paused: None,
        });
        self.streams.len() as u32 - 1
    }

    /// Builds the lanes (`TrajectorySource`) and sends the warmup.
    fn start_lanes(&mut self) -> Result<(), String> {
        for lane in 0..self.cfg.concurrency {
            let (trace, plan) = self.next_trace()?;
            let t_star = snapshot::t_star(
                &plan,
                self.cfg.seed,
                lane,
                self.cfg.start_min_ratio,
                self.cfg.start_max_ratio,
            )
            .ok_or_else(|| format!("trace {} has no timestamps", plan.trace_id))?;
            let snap = snapshot::snapshot(&plan, t_star).ok_or_else(|| {
                format!("trace {}: nothing live at t* = {t_star} ms", plan.trace_id)
            })?;
            let play = self.new_play(trace, plan.clone(), lane)?;
            // One stream per live state; the warmup primes their history.
            let mut streams = Vec::with_capacity(snap.states.len());
            for state in &snap.states {
                let stream = self.new_stream(play, state.conv, state.depth);
                if state.depth == 0 {
                    self.plays[play as usize].root = Some(stream);
                }
                streams.push(stream);
            }
            self.lanes.push(Lane {
                play,
                snapshot: snap,
                streams,
            });
        }
        let mut warm: Vec<(u32, u32, f64)> = Vec::new();
        for lane in &self.lanes {
            let plan = &self.plays[lane.play as usize].plan;
            for (state, &stream) in lane.snapshot.states.iter().zip(&lane.streams) {
                if let Some(turn) = state.warmup_turn() {
                    let ts = plan.convs[state.conv as usize].turns[turn as usize].ts_ms;
                    warm.push((stream, turn, lane.snapshot.t_star_ms - ts));
                }
            }
        }
        self.warmup_outstanding = warm.len();
        self.stats.warmup_requests = warm.len() as u64;
        if warm.is_empty() {
            return self.start_profiling();
        }
        let max_lead = warm.iter().map(|w| w.2).fold(0.0, f64::max);
        for (stream, turn, lead) in warm {
            if self.cfg.burst_starts || max_lead - lead <= 0.0 {
                self.issue(stream, turn, true, None);
            } else {
                self.queue
                    .timer(ns(max_lead - lead), Event::Warm { stream, turn });
            }
        }
        Ok(())
    }

    fn start_profiling(&mut self) -> Result<(), String> {
        self.phase = Phase::Profiling;
        self.profiling_start = self.now;
        self.working_set.start(self.now);
        self.profiling_end = self.now + ns(self.cfg.duration_s * 1e3);
        self.grace_end = self.profiling_end + ns(self.cfg.grace_s * 1e3);
        self.stats.profiling_start_s = self.now as f64 / 1e9;
        // Seed every lane's barrier history and live-children joins before any lane dispatches.
        let mut dispatches = Vec::new();
        for lane in 0..self.lanes.len() {
            let play = self.lanes[lane].play;
            let snap = self.lanes[lane].snapshot.clone();
            let streams = self.lanes[lane].streams.clone();
            let plan = self.plays[play as usize].plan.clone();
            for &(conv, done) in &snap.boundaries {
                for turn in 0..done {
                    self.plays[play as usize].completed.insert((conv, turn));
                }
            }
            let root_state = snap.states.iter().find(|s| s.depth == 0).cloned();
            self.plays[play as usize].root_pending = root_state.is_some();
            for (state, &stream) in snap.states.iter().zip(&streams) {
                if state.depth == 0 {
                    continue;
                }
                self.plays[play as usize].descendants += 1;
                let (Some(root), Some(gated), Some(branch)) =
                    (&root_state, state.join_turn, state.branch)
                else {
                    continue;
                };
                let gate = self.plays[play as usize].seeded_gate(root, gated, &plan);
                if let Some(p) = gate.prereq(branch) {
                    p.expected += 1;
                    p.registered = true;
                }
                self.streams[stream as usize].joins.push((gated, branch));
            }
            let dispatchable: Vec<(usize, f64)> = snap
                .states
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.waiting)
                .map(|(i, s)| (i, s.offset_ms))
                .collect();
            let t0 = if self.cfg.burst_starts {
                dispatchable
                    .iter()
                    .map(|d| d.1)
                    .fold(f64::INFINITY, f64::min)
            } else {
                0.0
            };
            for (i, offset) in dispatchable {
                let state = &snap.states[i];
                let kind = if state.depth == 0 {
                    Kind::SessionStart
                } else {
                    Kind::ChildNext
                };
                dispatches.push((streams[i], state.next_turn, kind, offset - t0));
            }
        }
        for (stream, turn, kind, delay) in dispatches {
            if delay > 0.0 {
                self.queue.timer(
                    self.now + ns(delay),
                    Event::Dispatch {
                        stream,
                        turn,
                        kind,
                        delay_ms: None,
                    },
                );
            } else {
                self.submit(stream, turn, kind, None);
            }
        }
        self.enforce_idle_cap(false);
        Ok(())
    }

    /// `_spawn_children_and_register_gates`.
    fn spawn_children(&mut self, play: u32, branches: &[u32], origin: Option<f64>) {
        let plan = self.plays[play as usize].plan.clone();
        let mut immediate = Vec::new();
        for &b in branches {
            if origin.is_none() && self.plays[play as usize].overlap_spawned.contains(&b) {
                continue;
            }
            let branch = &plan.root().branches[b as usize];
            let gated = if branch.background {
                None
            } else {
                branch.join_turn
            };
            let base = origin.unwrap_or(branch.start_ms);
            for &child in &branch.children {
                let stream = self.new_stream(play, child, 1);
                if let Some(g) = gated {
                    if let Some(p) = self.plays[play as usize].future_gate(g).prereq(b) {
                        p.expected += 1;
                        p.registered = true;
                    }
                    self.streams[stream as usize].joins.push((g, b));
                }
                self.plays[play as usize].descendants += 1;
                let first = plan.convs[child as usize].turns.first();
                let offset = first.map_or(0.0, |t| (t.ts_ms - base).max(0.0));
                if offset > 0.0 {
                    let at = self.now + ns(offset);
                    self.queue.fixed(
                        at,
                        Event::Dispatch {
                            stream,
                            turn: 0,
                            kind: Kind::ChildFirst,
                            delay_ms: None,
                        },
                    );
                } else {
                    immediate.push(stream);
                }
            }
        }
        for stream in immediate {
            self.submit(stream, 0, Kind::ChildFirst, None);
        }
    }

    /// Sends a turn through the replay barrier: held until its predecessors have completed.
    fn submit(&mut self, stream: u32, turn: u32, kind: Kind, delay_ms: Option<f64>) {
        if self.phase == Phase::Profiling {
            let s = &self.streams[stream as usize];
            if self.plays[s.play as usize].hold(s.conv, turn, stream, kind) {
                self.stats.barrier_holds += 1;
                return;
            }
        }
        self.issue_ready(stream, turn, kind, delay_ms);
    }

    /// Sends a turn whose predecessors have completed, unless the duration is over. A refused child
    /// can never continue, so its joins are released (`on_child_stopped`).
    fn issue_ready(&mut self, stream: u32, turn: u32, kind: Kind, delay_ms: Option<f64>) {
        if self.sending_open() {
            self.issue(stream, turn, false, delay_ms);
        } else if kind.is_child() {
            self.child_done(stream);
        }
    }

    fn issue(&mut self, stream: u32, turn: u32, warmup: bool, delay_ms: Option<f64>) {
        let (play_id, conv, depth) = {
            let s = &self.streams[stream as usize];
            (s.play, s.conv, s.depth)
        };
        let play = &self.plays[play_id as usize];
        let plan = play.plan.clone();
        let t = &plan.convs[conv as usize].turns[turn as usize];
        let scale = self.cfg.token_scale.max(1);
        let tokens = Tokens::new(plan.block_size * scale, self.cfg.block_tokens);
        let prompt_tokens = tokens.prompt_tokens(t.hash_blocks);
        let mut prompt =
            Vec::with_capacity((prompt_tokens / self.cfg.block_tokens as u64) as usize);
        plan.prefixes.engine_blocks(
            t.prompt_end,
            tokens,
            play.namespace,
            &mut self.scratch,
            &mut prompt,
        );
        let output_tokens = if warmup {
            1
        } else {
            t.max_tokens as u64 * scale as u64
        };
        self.pause_ends(stream);
        self.set_context(stream, &prompt);
        let id = self.next_request;
        self.next_request += 1;
        self.idle_since = None;
        let req = EngineRequest {
            prompt: prompt.clone(),
            prompt_tokens,
            output_tokens,
        };
        self.in_flight.insert(
            id,
            InFlight {
                stream,
                turn,
                warmup,
                issued: self.now,
                admitted: 0,
                blocked: 0,
                first_token: 0,
                prompt,
                prompt_tokens,
                output_tokens,
                delay_ms,
                admission: None,
                recomputed: 0,
                timeline: None,
            },
        );
        match &mut self.scheduler {
            None => {
                if let Some(adm) = self.engine.submit(id, req) {
                    self.admitted(id, adm, 0);
                }
            }
            Some(s) => {
                if !self.engine.reject_oversized(&req) {
                    let at = self.now + s.host_ns(prompt_tokens, self.now);
                    self.queue.fixed(at, Event::Arrive(id));
                }
            }
        }
        if !warmup && depth == 0 {
            self.on_root_issued(play_id, turn);
        }
    }

    /// Contention: request `id` reaches the scheduler, which runs at once if it is idle.
    fn arrive(&mut self, id: u64) -> Result<(), String> {
        let f = &self.in_flight[&id];
        let req = EngineRequest {
            prompt: f.prompt.clone(),
            prompt_tokens: f.prompt_tokens,
            output_tokens: f.output_tokens,
        };
        let s = self
            .scheduler
            .as_mut()
            .expect("arrivals happen under contention");
        s.arrive(id, req);
        if s.busy() {
            return Ok(());
        }
        self.run_steps()
    }

    /// Contention: runs the step loop now and applies what happened.
    fn run_steps(&mut self) -> Result<(), String> {
        let mut out = std::mem::take(&mut self.outcomes);
        let next = self
            .scheduler
            .as_mut()
            .expect("steps run under contention")
            .run(self.now, &mut self.engine, &mut out);
        for outcome in out.drain(..) {
            match outcome {
                Outcome::Admitted {
                    id,
                    admission,
                    blocked_ns,
                } => self.admitted(id, admission, blocked_ns),
                Outcome::FirstToken(id) => {
                    self.in_flight
                        .get_mut(&id)
                        .expect("a request with a first token is in flight")
                        .first_token = self.now;
                }
                Outcome::Finished(id) => self.complete(id)?,
            }
        }
        self.outcomes = out;
        self.step_at = next;
        if let Some(at) = next {
            self.queue.fixed(at, Event::Step);
        }
        Ok(())
    }

    fn admitted(&mut self, id: u64, adm: Admission, blocked: u64) {
        let now = self.now;
        let f = self
            .in_flight
            .get_mut(&id)
            .expect("admitted request is in flight");
        f.admitted = now;
        f.blocked = blocked;
        f.admission = Some(adm);
        f.recomputed =
            recomputed_tokens(&mut self.seen, &f.prompt, adm, self.cfg.block_tokens as u64);
        let done = self
            .scheduler
            .is_none()
            .then(|| now + self.cost.service(adm.sources, f.output_tokens).as_nanos() as u64);
        if let Some(tl) = &mut self.timeline {
            let stream = &self.streams[f.stream as usize];
            let play = &self.plays[stream.play as usize];
            f.timeline = Some(tl.requests.len());
            tl.events.push(TimelineEvent::Admit(tl.requests.len()));
            tl.requests.push(TimelineRequest {
                prompt: f.prompt.clone(),
                prompt_tokens: f.prompt_tokens,
                output_tokens: f.output_tokens,
                record: None,
                warmup: f.warmup,
                trace: play.trace,
                conv: stream.conv,
                turn: f.turn,
                play: play.namespace,
            });
        }
        let (stream, turn) = (f.stream, f.turn);
        if let Some(done) = done {
            self.queue.fixed(done, Event::Complete(id));
        }
        self.pause_admitted(id, stream, turn);
    }

    /// Request `id` returned its last token. Under contention the scheduler has already finished it
    /// in the engine.
    fn complete(&mut self, id: u64) -> Result<(), String> {
        let f = self
            .in_flight
            .remove(&id)
            .expect("completion of a request in flight");
        if let (Some(tl), Some(i)) = (&mut self.timeline, f.timeline) {
            tl.events.push(TimelineEvent::Finish(i));
        }
        if self.scheduler.is_none() {
            // The idle engine queues a request only while it waits for blocks.
            for (qid, adm) in self.engine.finish(id, self.now) {
                let blocked = self.now - self.in_flight[&qid].issued;
                self.admitted(qid, adm, blocked);
            }
        }
        let s = &self.streams[f.stream as usize];
        let plan = self.plays[s.play as usize].plan.clone();
        let conv = &plan.convs[s.conv as usize];
        let adm = f.admission.expect("a completed request was admitted");
        let counted =
            !f.warmup && self.phase == Phase::Profiling && f.issued >= self.profiling_start;
        if counted {
            if let (Some(tl), Some(i)) = (&mut self.timeline, f.timeline) {
                tl.requests[i].record = Some(self.records.len());
            }
            let ttft = match self.scheduler {
                None => f.admitted - f.issued + self.cost.ttft(adm.sources).as_nanos() as u64,
                Some(_) => f.first_token - f.issued,
            };
            self.records.push(RequestRecord {
                trace: self.plays[s.play as usize].trace,
                conv: s.conv,
                turn: f.turn,
                kind: conv.kind,
                warmup: false,
                prompt_tokens: f.prompt_tokens,
                output_tokens: f.output_tokens,
                sources: adm.sources,
                recomputed: f.recomputed,
                delay_ms: f.delay_ms,
                issued_ns: f.issued - self.profiling_start,
                admitted_ns: f.admitted - self.profiling_start,
                blocked_ns: f.blocked,
                done_ns: self.now - self.profiling_start,
                ttft_ns: ttft,
            });
        }
        if f.warmup {
            self.warmup_outstanding -= 1;
            if self.warmup_outstanding == 0 {
                self.start_profiling()?;
            }
            return Ok(());
        }
        self.credit_return(f.stream, f.turn);
        self.enforce_idle_cap(true);
        Ok(())
    }

    /// The return path of one request (`CreditCallbackHandler.on_credit_return`).
    fn credit_return(&mut self, stream: u32, turn: u32) {
        let (play, conv, depth) = {
            let s = &self.streams[stream as usize];
            (s.play, s.conv, s.depth)
        };
        let plan = self.plays[play as usize].plan.clone();
        let is_final = turn as usize + 1 == plan.convs[conv as usize].turns.len();
        if depth > 0 && is_final {
            self.child_done(stream);
        }
        self.plays[play as usize].completed.insert((conv, turn));
        self.release_held(play);
        let mut intercepted = false;
        if depth == 0 {
            let spawns = plan.convs[conv as usize].turns[turn as usize]
                .spawns
                .clone();
            if !spawns.is_empty() {
                self.spawn_children(play, &spawns, None);
            }
            intercepted = self.plays[play as usize].suspend_if_gated(turn + 1);
            if is_final {
                self.plays[play as usize].root_pending = false;
                self.maybe_drain(play);
            }
        }
        if intercepted || is_final {
            return;
        }
        if !self.sending_open() {
            if depth > 0 {
                self.child_done(stream);
            }
            return;
        }
        let next = turn + 1;
        let (delay, tool) = self.delay(stream, next);
        let kind = if depth == 0 {
            Kind::RootNext
        } else {
            Kind::ChildNext
        };
        match delay {
            Some(d) if d > 0.0 => {
                let at = self.now + ns(d);
                self.queue.timer(
                    at,
                    Event::Dispatch {
                        stream,
                        turn: next,
                        kind,
                        delay_ms: Some(d),
                    },
                );
                self.pause(stream, at, tool);
            }
            _ => self.submit(stream, next, kind, delay),
        }
    }

    /// The delay before turn `turn` of `stream` and the tool it waits for: the trace's, or a
    /// TraceLab draw in its place (the same for every arm).
    fn delay(&self, stream: u32, turn: u32) -> (Option<f64>, Option<u16>) {
        let s = &self.streams[stream as usize];
        let play = &self.plays[s.play as usize];
        let delay = play.plan.convs[s.conv as usize].turns[turn as usize].delay_ms;
        match (self.cfg.pauses, delay) {
            (Pauses::TraceLab { cap_s }, Some(d)) if d > 0.0 => {
                let key = self.cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (u64::from(play.trace) << 40)
                    ^ (u64::from(s.conv) << 20)
                    ^ u64::from(turn);
                let (tool, ms) = ToolPauses::tracelab().draw(key);
                (Some(cap_s.map_or(ms, |cap| ms.min(cap * 1e3))), Some(tool))
            }
            _ => (delay, None),
        }
    }

    /// A child stream finished (or can never continue): satisfy its gates and drain.
    fn child_done(&mut self, stream: u32) {
        let s = &mut self.streams[stream as usize];
        if s.done {
            return;
        }
        s.done = true;
        let play = s.play;
        let joins = std::mem::take(&mut s.joins);
        self.clear_context(stream);
        for (gated, branch) in joins {
            if let Some(root) = self.plays[play as usize].satisfy(gated, branch) {
                self.submit(root, gated, Kind::Join, None);
            }
        }
        let p = &mut self.plays[play as usize];
        p.descendants = p.descendants.saturating_sub(1);
        self.maybe_drain(play);
    }

    /// A session tree that has finished frees its lane, which starts the next trace.
    fn maybe_drain(&mut self, play: u32) {
        let p = &mut self.plays[play as usize];
        if !p.drain() {
            return;
        }
        let lane = p.lane;
        if let Some(root) = p.root {
            self.clear_context(root);
        }
        self.queue.timer(self.now, Event::Recycle(lane));
    }

    fn recycle(&mut self, lane: usize) -> Result<(), String> {
        if !self.can_start_session() {
            return Ok(());
        }
        let (trace, plan) = self.next_trace()?;
        let play = self.new_play(trace, plan, lane)?;
        let root = self.new_stream(play, 0, 0);
        let p = &mut self.plays[play as usize];
        p.root = Some(root);
        p.root_pending = true;
        self.submit(root, 0, Kind::SessionStart, None);
        Ok(())
    }

    /// `on_credit_issued`: a root turn whose recorded spawn overlapped the request starts those
    /// branches now, offset from the turn's recorded start.
    fn on_root_issued(&mut self, play: u32, turn: u32) {
        let plan = self.plays[play as usize].plan.clone();
        let t = &plan.root().turns[turn as usize];
        let Some(api) = t.api_ms.filter(|a| *a > 0.0) else {
            return;
        };
        let end = t.ts_ms + api;
        let overlapping: Vec<u32> = t
            .spawns
            .iter()
            .copied()
            .filter(|&b| plan.root().branches[b as usize].start_ms < end)
            .collect();
        if overlapping.is_empty() {
            return;
        }
        self.spawn_children(play, &overlapping, Some(t.ts_ms));
        self.plays[play as usize]
            .overlap_spawned
            .extend(overlapping);
    }

    /// Sends the turns of `play` that its barriers held and that may go now.
    fn release_held(&mut self, play: u32) {
        for key in self.plays[play as usize].unblocked() {
            let (stream, kind) = self.plays[play as usize]
                .held
                .remove(&key)
                .expect("collected above");
            self.issue_ready(stream, key.1, kind, None);
        }
    }

    /// `enforce_system_idle_cap`: with nothing in flight, pull every pending scheduler timer, and the
    /// timers that follow them, forward by the same amount so the next scheduler timer fires within
    /// the cap.
    fn enforce_idle_cap(&mut self, follows_return: bool) {
        let Some(cap) = self.cfg.system_idle_cap_s else {
            return;
        };
        if !self.in_flight.is_empty() {
            self.idle_since = None;
            return;
        }
        if follows_return || self.idle_since.is_none() {
            self.idle_since = Some(self.now);
        }
        let elapsed = self.now - self.idle_since.expect("set above");
        let remaining = ns(cap * 1e3).saturating_sub(elapsed);
        // The grace period sends nothing, so what the cap skips then is not counted.
        if let Some(jump) = self.queue.pull_forward(self.now + remaining)
            && self.now < self.profiling_end
        {
            self.stats.idle_cap_jumps += 1;
            self.stats.idle_cap_skipped_s += jump as f64 / 1e9;
        }
    }

    /// Replaces a stream's context (its latest prompt) in the live working set.
    fn set_context(&mut self, stream: u32, prompt: &[BlockId]) {
        self.working_set.add(prompt);
        self.clear_context(stream);
        self.streams[stream as usize].context = prompt.to_vec();
        if self.phase == Phase::Profiling {
            self.working_set.observe_peak();
        }
    }

    /// Takes a stream's latest prompt out of the live working set.
    fn clear_context(&mut self, stream: u32) {
        let context = std::mem::take(&mut self.streams[stream as usize].context);
        self.working_set.remove(&context);
    }
}
