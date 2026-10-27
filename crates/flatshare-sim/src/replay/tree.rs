//! AIPerf's session trees as the replay plays them (`session_tree.py`, `replay_dependencies.py`): a play of one
//! trace, its streams (the root conversation and the children its branches spawn), the gates on which a root turn
//! waits for the children it joins, and the barriers that hold a turn until its replay predecessors complete.

use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;

use crate::aiperf::{ReplayPlan, Snapshot, snapshot};
use crate::prefix::BlockId;

use super::retain::Paused;

/// How a turn came to be sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    SessionStart,
    RootNext,
    Join,
    ChildFirst,
    ChildNext,
}

impl Kind {
    pub fn is_child(self) -> bool {
        matches!(self, Kind::ChildFirst | Kind::ChildNext)
    }
}

/// One conversation of a play as it is replayed: the root, or a child spawned by a branch.
#[derive(Debug)]
pub(super) struct Stream {
    pub play: u32,
    pub conv: u32,
    /// 0 for the root, 1 for a child.
    pub depth: u8,
    /// `(gated root turn, branch)` this child must satisfy; empty for background children.
    pub joins: Vec<(u32, u32)>,
    /// A child that finished, or that can never continue once sending stops.
    pub done: bool,
    /// The blocks of its latest prompt, counted in the live working set.
    pub context: Vec<BlockId>,
    /// Between turns, with a retention arm.
    pub paused: Option<Paused>,
}

/// One branch a gated root turn waits for.
#[derive(Debug)]
pub(super) struct Prereq {
    branch: u32,
    /// Children spawned so far.
    pub expected: u32,
    /// Children finished so far.
    completed: u32,
    /// The branch has been spawned, so `expected` is final.
    pub registered: bool,
}

/// A root turn that waits for the children of the branches it joins (AIPerf's SPAWN_JOIN).
#[derive(Debug)]
pub(super) struct Gate {
    turn: u32,
    prereqs: Vec<Prereq>,
}

impl Gate {
    fn satisfied(&self) -> bool {
        self.prereqs
            .iter()
            .all(|p| p.registered && p.completed >= p.expected)
    }

    pub fn prereq(&mut self, branch: u32) -> Option<&mut Prereq> {
        self.prereqs.iter_mut().find(|p| p.branch == branch)
    }
}

/// One session play: a trace replayed by a lane, under its own KV namespace.
#[derive(Debug)]
pub(super) struct Play {
    pub plan: Rc<ReplayPlan>,
    /// The trace's index in the corpus.
    pub trace: u32,
    pub lane: usize,
    /// The KV namespace its blocks are cut in: the play's index.
    pub namespace: u64,
    pub root: Option<u32>,
    /// The root has turns left to send.
    pub root_pending: bool,
    /// Child streams not yet done.
    pub descendants: u32,
    /// The session tree has finished and its lane has moved on.
    drained: bool,
    /// `(conversation, turn)` pairs completed: the replay barriers' history.
    pub completed: HashSet<(u32, u32)>,
    /// Turns held by replay barriers, in AIPerf's `(conversation id, turn)` order, with their stream and
    /// kind.
    pub held: BTreeMap<(u32, u32, u32), (u32, Kind)>,
    /// Gates of later root turns.
    future_joins: BTreeMap<u32, Gate>,
    /// The gate the root is blocked on now.
    active_join: Option<Gate>,
    /// Branches spawned when their spawning turn was sent, so its return does not spawn them again.
    pub overlap_spawned: Vec<u32>,
}

impl Play {
    /// A play of the trace at index `trace` of the corpus by `lane`, in KV namespace `namespace`, with no stream yet.
    pub fn new(plan: Rc<ReplayPlan>, trace: u32, lane: usize, namespace: u64) -> Self {
        Self {
            plan,
            trace,
            lane,
            namespace,
            root: None,
            root_pending: false,
            descendants: 0,
            drained: false,
            completed: HashSet::new(),
            held: BTreeMap::new(),
            future_joins: BTreeMap::new(),
            active_join: None,
            overlap_spawned: Vec::new(),
        }
    }

    /// `_ensure_seeded_join`: the join on `gated` for a snapshot lane, created with the prerequisites
    /// of every branch the turn joins; branches spawned before `t*` count as registered.
    pub fn seeded_gate(
        &mut self,
        root: &snapshot::State,
        gated: u32,
        plan: &ReplayPlan,
    ) -> &mut Gate {
        if self.active_join.as_ref().is_some_and(|g| g.turn == gated) {
            return self.active_join.as_mut().expect("checked");
        }
        if !self.future_joins.contains_key(&gated) {
            let prereqs = plan.root().turns[gated as usize]
                .joins
                .iter()
                .map(|&b| Prereq {
                    branch: b,
                    expected: 0,
                    completed: 0,
                    registered: plan.root().branches[b as usize].spawn_turn < root.next_turn,
                })
                .collect();
            let gate = Gate {
                turn: gated,
                prereqs,
            };
            if root.waiting && root.join_turn == Some(gated) {
                self.active_join = Some(gate);
                return self.active_join.as_mut().expect("just set");
            }
            self.future_joins.insert(gated, gate);
        }
        self.future_joins.get_mut(&gated).expect("present")
    }

    /// `_ensure_future_join`.
    pub fn future_gate(&mut self, gated: u32) -> &mut Gate {
        let plan = self.plan.clone();
        self.future_joins.entry(gated).or_insert_with(|| Gate {
            turn: gated,
            prereqs: plan.root().turns[gated as usize]
                .joins
                .iter()
                .map(|&b| Prereq {
                    branch: b,
                    expected: 0,
                    completed: 0,
                    registered: false,
                })
                .collect(),
        })
    }

    /// `_maybe_suspend_parent`: whether the root must wait for children before `next`.
    pub fn suspend_if_gated(&mut self, next: u32) -> bool {
        if let Some(active) = &self.active_join
            && active.turn == next
            && !active.satisfied()
        {
            return true;
        }
        match self.future_joins.get(&next) {
            None => false,
            Some(g) if g.satisfied() => {
                self.future_joins.remove(&next);
                false
            }
            Some(_) => {
                self.active_join = self.future_joins.remove(&next);
                true
            }
        }
    }

    /// `_satisfy_prerequisite`: counts one finished child against the join on `gated`. Returns the
    /// root stream when the root was blocked on that join and may now send it.
    pub fn satisfy(&mut self, gated: u32, branch: u32) -> Option<u32> {
        let active = self.active_join.as_ref().is_some_and(|g| g.turn == gated);
        let gate = if active {
            self.active_join.as_mut()?
        } else {
            self.future_joins.get_mut(&gated)?
        };
        gate.prereq(branch)?.completed += 1;
        if !gate.satisfied() {
            return None;
        }
        if active {
            self.active_join = None;
            self.root
        } else {
            self.future_joins.remove(&gated);
            None
        }
    }

    /// Holds turn `turn` of conversation `conv`, sent by `stream` as `kind`, at its replay barrier unless every
    /// predecessor has completed. Returns whether it holds it.
    pub fn hold(&mut self, conv: u32, turn: u32, stream: u32, kind: Kind) -> bool {
        let t = &self.plan.convs[conv as usize].turns[turn as usize];
        if t.preds.iter().all(|p| self.completed.contains(p)) {
            return false;
        }
        let key = (self.plan.id_rank[conv as usize], turn, conv);
        self.held.insert(key, (stream, kind));
        true
    }

    /// The held turns whose predecessors have all completed, in AIPerf's order.
    pub fn unblocked(&self) -> Vec<(u32, u32, u32)> {
        self.held
            .keys()
            .filter(|&&(_, turn, conv)| {
                self.plan.convs[conv as usize].turns[turn as usize]
                    .preds
                    .iter()
                    .all(|x| self.completed.contains(x))
            })
            .copied()
            .collect()
    }

    /// Whether the session tree has just finished: its root has no turn left to send and no child stream is
    /// running. It is then drained, once, and forgets its barriers.
    pub fn drain(&mut self) -> bool {
        if self.drained || self.root_pending || self.descendants > 0 {
            return false;
        }
        self.drained = true;
        self.completed.clear();
        self.held.clear();
        true
    }
}

/// A lane's first session: its play, its snapshot, and one stream per live state.
#[derive(Debug)]
pub(super) struct Lane {
    pub play: u32,
    pub snapshot: Snapshot,
    pub streams: Vec<u32>,
}
