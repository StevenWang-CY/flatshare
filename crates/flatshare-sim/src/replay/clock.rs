//! The replay's pending events, and how far the idle cap has pulled them forward.
//!
//! Events wait in three queues, by what AIPerf's idle cap (`systemIdleGapCapSeconds`) does to them:
//! - **timers**: the scheduler's (spread warmup, first dispatches, think times, recycling), which the cap pulls
//!   forward while nothing is in flight;
//! - **following**: timers the cap pulls forward with the scheduler's but leaves out of its test for the next one:
//!   the oracle's prefetches, which follow the dispatch they foresee;
//! - **fixed**: events on the engine's clock, which the cap leaves in place: request arrivals, steps and
//!   completions, children's first turns, and the other retention timers.
//!
//! The first two queues store each event at the time it was due when pushed plus the shift so far, so that pulling
//! them forward is one addition to the shift: an event stored at `at` fires at `at - shift`. Events due at the same
//! time fire in the order they were pushed, whichever queue holds them.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// An event waiting in a queue: when it is due as stored, and its place in the order of pushes.
#[derive(Debug)]
struct Entry<E> {
    at: u64,
    seq: u64,
    event: E,
}

impl<E> PartialEq for Entry<E> {
    fn eq(&self, other: &Self) -> bool {
        (self.at, self.seq) == (other.at, other.seq)
    }
}

impl<E> Eq for Entry<E> {}

impl<E> PartialOrd for Entry<E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E> Ord for Entry<E> {
    // Reversed: BinaryHeap pops the earliest.
    fn cmp(&self, other: &Self) -> Ordering {
        (other.at, other.seq).cmp(&(self.at, self.seq))
    }
}

/// The pending events of a replay.
#[derive(Debug)]
pub(super) struct Queue<E> {
    /// Pushes so far: breaks ties between events due at the same time.
    seq: u64,
    /// How far the idle cap has pulled `timers` and `following` forward.
    shift: u64,
    timers: BinaryHeap<Entry<E>>,
    following: BinaryHeap<Entry<E>>,
    fixed: BinaryHeap<Entry<E>>,
}

impl<E> Default for Queue<E> {
    fn default() -> Self {
        Self {
            seq: 0,
            shift: 0,
            timers: BinaryHeap::new(),
            following: BinaryHeap::new(),
            fixed: BinaryHeap::new(),
        }
    }
}

impl<E> Queue<E> {
    /// A scheduler timer due at `at`.
    pub fn timer(&mut self, at: u64, event: E) {
        self.seq += 1;
        self.timers.push(Entry {
            at: at + self.shift,
            seq: self.seq,
            event,
        });
    }

    /// A timer that follows the scheduler's, due at `at`.
    pub fn following(&mut self, at: u64, event: E) {
        self.seq += 1;
        self.following.push(Entry {
            at: at + self.shift,
            seq: self.seq,
            event,
        });
    }

    /// An event on the engine's clock, due at `at`.
    pub fn fixed(&mut self, at: u64, event: E) {
        self.seq += 1;
        self.fixed.push(Entry {
            at,
            seq: self.seq,
            event,
        });
    }

    /// Takes the next event at `now`: when it is due, the event, and whether it is a scheduler timer. A following
    /// timer the cap pulled into the past is due at once.
    pub fn pop(&mut self, now: u64) -> Option<(u64, E, bool)> {
        let shift = self.shift;
        let heads = [
            self.timers.peek().map(|e| (e.at - shift, e.seq)),
            self.following
                .peek()
                .map(|e| (e.at.saturating_sub(shift).max(now), e.seq)),
            self.fixed.peek().map(|e| (e.at, e.seq)),
        ];
        let (queue, (at, _)) = heads
            .into_iter()
            .enumerate()
            .filter_map(|(i, head)| Some((i, head?)))
            .min_by_key(|&(_, head)| head)?;
        let entry = match queue {
            0 => self.timers.pop(),
            1 => self.following.pop(),
            _ => self.fixed.pop(),
        }
        .expect("peeked");
        Some((at, entry.event, queue == 0))
    }

    /// Pulls the scheduler's timers, and those that follow them, forward by as much as makes the next scheduler
    /// timer due by `limit`. Returns how far, if that timer was due later.
    pub fn pull_forward(&mut self, limit: u64) -> Option<u64> {
        let earliest = self.timers.peek()?.at - self.shift;
        let jump = earliest.checked_sub(limit).filter(|&jump| jump > 0)?;
        self.shift += jump;
        Some(jump)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_fire_by_time_then_push_order_across_the_queues() {
        let mut queue = Queue::default();
        queue.fixed(5, "fixed at 5");
        queue.timer(5, "timer at 5");
        queue.following(3, "following at 3");
        queue.timer(1, "timer at 1");
        let order: Vec<_> = std::iter::from_fn(|| queue.pop(0)).collect();
        assert_eq!(
            order,
            [
                (1, "timer at 1", true),
                (3, "following at 3", false),
                (5, "fixed at 5", false),
                (5, "timer at 5", true),
            ]
        );
    }

    #[test]
    fn the_cap_pulls_the_timers_forward_and_leaves_the_engines_clock() {
        let mut queue = Queue::default();
        queue.timer(100, "think time");
        queue.following(90, "prefetch");
        queue.fixed(50, "step");
        // Nothing to pull while the next timer is due within the limit.
        assert_eq!(queue.pull_forward(100), None);
        assert_eq!(queue.pull_forward(30), Some(70));
        // A timer pushed after the jump is due when it says.
        queue.timer(40, "dispatch");
        // The prefetch, pulled 70 into the past of now = 25, is due at once.
        assert_eq!(queue.pop(25), Some((25, "prefetch", false)));
        assert_eq!(queue.pop(25), Some((30, "think time", true)));
        assert_eq!(queue.pop(30), Some((40, "dispatch", true)));
        assert_eq!(queue.pop(40), Some((50, "step", false)));
        assert_eq!(queue.pop(50), None);
        assert_eq!(queue.pull_forward(0), None, "no timer left");
    }
}
