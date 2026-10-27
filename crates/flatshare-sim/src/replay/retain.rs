//! What a [`Retention`] arm does with a session's KV while the session pauses between its turns
//! ([`crate::retention`]): from when a turn returns to when its successor, due after a delay, is admitted.
//!
//! At a pause the arm may hold the session's cached blocks in the pool (the TTLs, and InferCept when it preserves),
//! demote them to be reused first (Tokencake, and InferCept otherwise), or start a load of the blocks the CPU tier
//! holds so that it ends just before the session is due back, by the history's prediction or the oracle's exact
//! time, and hold what it loads (the prefetch arms and Tokencake). A hold ends when the next turn is admitted, at
//! its TTL, or once the session is later than twice its predicted pause; the engine also releases holds for a
//! request that needs their blocks. Under MORI an admitted request's blocks rank by when its session is predicted
//! back.

use super::{Event, Replay, ns};
use crate::retention::{Interception, Retention, intercept};

/// A retention timer of a pause.
#[derive(Clone, Copy, Debug)]
pub(super) enum RetainStep {
    /// The pause's hold ends: a TTL, or the session is late.
    Release,
    /// Start loading the session's blocks from the CPU tier.
    Prefetch,
    /// The load has ended: the blocks are in the pool, held for the session.
    Prefetched,
}

/// A session between its turns, until its next turn is admitted.
#[derive(Debug)]
pub(super) struct Paused {
    id: u64,
    since: u64,
    /// The tool it waits for, when the pauses name one.
    tool: Option<u16>,
    /// The engine's hold on its KV.
    hold: Option<u64>,
}

impl Replay<'_> {
    /// What the history level predicts for the pause before turn `turn` of `stream`, in ms.
    fn predicted_pause(&self, stream: u32, turn: u32) -> Option<f64> {
        let s = &self.streams[stream as usize];
        let turns = self.plays[s.play as usize].plan.convs[s.conv as usize]
            .turns
            .len();
        let tool = if (turn as usize) < turns {
            self.delay(stream, turn).1
        } else {
            None
        };
        self.history.predict(stream, tool)
    }

    /// `stream` pauses until its next turn, due at `at`, after calling `tool`: the retention arm
    /// acts on its KV, its latest prompt.
    pub(super) fn pause(&mut self, stream: u32, at: u64, tool: Option<u16>) {
        // A turn that was never admitted (larger than the pool) leaves its pause behind.
        if let Some(hold) = self.streams[stream as usize]
            .paused
            .take()
            .and_then(|p| p.hold)
        {
            self.release(hold);
        }
        self.next_pause += 1;
        let id = self.next_pause;
        let context = self.streams[stream as usize].context.clone();
        let tokens = context.len() as u64 * self.cfg.block_tokens as u64;
        let predicted = self.history.predict(stream, tool).map(ns);
        let mut hold = None;
        let mut back = None;
        match self.retention {
            Retention::None | Retention::Mori => {}
            Retention::Ttl(ttl_s) => {
                hold = self.engine.hold(&context);
                let end = self.now + ns(ttl_s * 1e3);
                self.retain_at(end, stream, id, RetainStep::Release);
            }
            Retention::HistoryPrefetch => back = predicted.map(|p| self.now + p),
            Retention::OraclePrefetch => back = Some(at.max(self.now)),
            Retention::Tokencake => {
                self.engine.demote(&context);
                back = predicted.map(|p| self.now + p);
            }
            Retention::InferCept => {
                let restore = self.cost.restore_ms(tokens);
                let choice = predicted.map(|p| {
                    intercept(
                        tokens,
                        p as f64 / 1e6,
                        self.engine.running_tokens(),
                        self.restore_stall * restore,
                        self.cost.recompute_ms(tokens),
                        self.engine.has_cpu_tier(),
                    )
                });
                if choice == Some(Interception::Preserve) {
                    hold = self.engine.hold(&context);
                    back = predicted.map(|p| self.now + p);
                } else {
                    self.engine.demote(&context);
                }
            }
        }
        if let Some(back) = back {
            if self.retention != Retention::InferCept {
                // Start the load so that it ends just before the session returns. The oracle knows
                // when the turn is sent, so its load moves with the dispatch when the idle cap pulls
                // that forward; the history level predicts the engine's time.
                let lead = ns(self.cost.restore_ms(tokens) + 1.0);
                let start = back.saturating_sub(lead).max(self.now);
                let prefetch = Event::Retain {
                    stream,
                    pause: id,
                    step: RetainStep::Prefetch,
                };
                if self.retention == Retention::OraclePrefetch {
                    self.queue.following(start, prefetch);
                } else {
                    self.queue.fixed(start, prefetch);
                }
            }
            // A session later than twice its expected pause is not coming back soon.
            let late = back + (back - self.now);
            self.retain_at(late, stream, id, RetainStep::Release);
        }
        self.streams[stream as usize].paused = Some(Paused {
            id,
            since: self.now,
            tool,
            hold,
        });
    }

    /// `stream` sends its next turn: the history learns how long it paused. A prefetch that lands
    /// before the turn is admitted still serves it.
    pub(super) fn pause_ends(&mut self, stream: u32) {
        if let Some(p) = &self.streams[stream as usize].paused {
            let ms = (self.now - p.since) as f64 / 1e6;
            self.history.observe(stream, p.tool, ms);
        }
    }

    /// Request `id`, turn `turn` of `stream`, was admitted: under MORI its blocks are ranked by when the session
    /// is expected back after it, and the pause before it ends, its hits pinned now, so its hold can go.
    pub(super) fn pause_admitted(&mut self, id: u64, stream: u32, turn: u32) {
        if self.retention == Retention::Mori
            && let Some(pause) = self.predicted_pause(stream, turn + 1)
        {
            self.engine.expect_pause(id, ns(pause));
        }
        if let Some(hold) = self.streams[stream as usize]
            .paused
            .take()
            .and_then(|p| p.hold)
        {
            self.release(hold);
        }
    }

    fn retain_at(&mut self, at: u64, stream: u32, pause: u64, step: RetainStep) {
        self.queue.fixed(
            at,
            Event::Retain {
                stream,
                pause,
                step,
            },
        );
    }

    /// A retention timer; stale once the turn after its pause has been admitted.
    pub(super) fn retain(
        &mut self,
        stream: u32,
        pause: u64,
        step: RetainStep,
    ) -> Result<(), String> {
        let s = stream as usize;
        if self.streams[s]
            .paused
            .as_ref()
            .is_none_or(|p| p.id != pause)
        {
            return Ok(());
        }
        match step {
            RetainStep::Release => {
                let hold = self.streams[s].paused.as_mut().and_then(|p| p.hold.take());
                if let Some(hold) = hold {
                    self.release(hold);
                    self.kick()?;
                }
            }
            RetainStep::Prefetch => {
                let context = &self.streams[s].context;
                let tokens =
                    self.engine.prefetchable(context) as u64 * self.cfg.block_tokens as u64;
                let end = match &mut self.scheduler {
                    Some(s) if tokens > 0 => s.load(self.now, tokens),
                    _ => self.now + ns(self.cost.restore_ms(tokens)),
                };
                self.retain_at(end, stream, pause, RetainStep::Prefetched);
            }
            RetainStep::Prefetched => {
                let stream = &mut self.streams[s];
                let hold = self.engine.prefetch(&stream.context);
                let p = stream.paused.as_mut().expect("checked above");
                debug_assert!(p.hold.is_none());
                p.hold = hold;
            }
        }
        Ok(())
    }

    /// Ends an engine hold; the idle engine admits what now fits.
    fn release(&mut self, hold: u64) {
        let now = self.now;
        for (qid, adm) in self.engine.release(hold) {
            let blocked = now - self.in_flight[&qid].issued;
            self.admitted(qid, adm, blocked);
        }
    }

    /// Contention: runs an idle step loop, which may now admit what waits for blocks.
    fn kick(&mut self) -> Result<(), String> {
        match &self.scheduler {
            Some(s) if !s.busy() => self.run_steps(),
            _ => Ok(()),
        }
    }
}
