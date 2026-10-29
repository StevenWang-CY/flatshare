//! The sessions placed on the nodes: each one's node, what each node holds, and the realtime sessions each node
//! carries for another. The table is the parked two-node design's, with call slots and moves.
//!
//! A session is keyed by its tenant and its id, since two trusted tenants may name sessions alike. At most
//! [`SESSIONS`] are remembered: a new session that finds the table full takes the place of the session idle longest,
//! and is placed without being remembered only while every session in the table has a request in flight. Untrusted
//! tenants' sessions, which each of their requests may name, take at most half the table, so that they cannot make it
//! forget the trusted tenants' idle sessions.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use flatshare_proto::{SessionId, SloClass};
use prometheus_client::metrics::gauge::Gauge;

/// Sessions remembered at most, a bound on what clients can make the gateway hold.
pub(crate) const SESSIONS: usize = 4096;

/// A session as the table keys it.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Key {
    /// Its tenant's index in the configuration.
    pub tenant: usize,
    pub session: SessionId,
}

/// A session's placement. Its requests hold it until they end, so a request of a session placed again meanwhile
/// ends on the placement it started on.
#[derive(Debug)]
pub(crate) struct Session {
    pub node: usize,
    /// The class it was placed with.
    pub class: SloClass,
    /// Whether a trusted tenant's request placed it.
    pub trusted: bool,
    /// The node it was moved from, for a realtime session moved because its node was not available: what makes its
    /// new node cover for a failed one.
    pub moved_from: Option<usize>,
    /// When it was placed.
    placed: Instant,
    /// Its requests in flight, and those waiting on their node.
    requests: AtomicU32,
    /// When its last request ended, in nanoseconds after it was placed (0 before any did).
    idle_from: AtomicU64,
}

impl Session {
    /// One of its requests began.
    pub fn began(&self) {
        self.requests.fetch_add(1, Ordering::AcqRel);
    }

    /// One of its requests ended at `now`.
    pub fn ended(&self, now: Instant) {
        let after = u64::try_from(now.saturating_duration_since(self.placed).as_nanos());
        self.idle_from
            .fetch_max(after.unwrap_or(u64::MAX), Ordering::Release);
        self.requests.fetch_sub(1, Ordering::AcqRel);
    }

    /// Whether it has a request in flight or waiting.
    fn busy(&self) -> bool {
        self.requests.load(Ordering::Acquire) > 0
    }

    /// When its idle time began: when its last request ended, or when it was placed.
    fn since(&self) -> Instant {
        self.placed + Duration::from_nanos(self.idle_from.load(Ordering::Acquire))
    }
}

/// What became of a session's placement.
#[derive(Debug)]
pub(crate) enum Found {
    /// It holds: its node is available.
    On(Arc<Session>),
    /// There is none: a new session, or one that idled out.
    None,
    /// Its node is not available.
    Lost(Arc<Session>),
}

/// The sessions placed on the nodes.
#[derive(Debug)]
pub(crate) struct Table {
    sessions: HashMap<Key, Arc<Session>>,
    /// Those of the sessions that untrusted tenants' requests placed: at most half the table.
    untrusted: usize,
    /// The sessions placed on each node by class, in the nodes' order, and their gauges.
    held: Vec<[u32; 3]>,
    gauges: Vec<[Gauge; 3]>,
    /// `moved_in[to][from]`: the realtime sessions on node `to` moved there from node `from`.
    moved_in: Vec<Vec<u32>>,
}

impl Table {
    /// An empty table for nodes whose session gauges are `gauges`, in order.
    pub fn new(gauges: Vec<[Gauge; 3]>) -> Self {
        let nodes = gauges.len();
        Self {
            sessions: HashMap::new(),
            untrusted: 0,
            held: vec![[0; 3]; nodes],
            gauges,
            moved_in: vec![vec![0; nodes]; nodes],
        }
    }

    /// The placement of `key`, if it holds on a node that is `available`.
    pub fn find(&self, key: &Key, available: impl Fn(usize) -> bool) -> Found {
        match self.sessions.get(key) {
            Some(placed) if available(placed.node) => Found::On(Arc::clone(placed)),
            Some(placed) => Found::Lost(Arc::clone(placed)),
            None => Found::None,
        }
    }

    /// The sessions placed on `node`, by class.
    pub fn held(&self, node: usize) -> [u32; 3] {
        self.held[node]
    }

    /// The agents' sessions placed on `node`, interactive and background: what its `max_agents` bounds.
    pub fn agents(&self, node: usize) -> u32 {
        let [_, interactive, background] = self.held[node];
        interactive + background
    }

    /// The nodes that `node` carries realtime sessions for, with how many.
    pub fn moved_in(&self, node: usize) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.moved_in[node]
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(from, n)| (from, *n))
    }

    /// Places `key` on `node`, replacing any placement it had, as a session of `class` of a `trusted` tenant or
    /// not, `moved_from` the node a realtime session left. `None` when the table has no room for it.
    pub fn place(
        &mut self,
        key: &Key,
        class: SloClass,
        node: usize,
        trusted: bool,
        moved_from: Option<usize>,
        now: Instant,
    ) -> Option<Arc<Session>> {
        if let Some(old) = self.sessions.remove(key) {
            self.release(&old);
        }
        if !self.make_room(trusted) {
            return None;
        }
        let moved_from = moved_from.filter(|from| class == SloClass::Realtime && *from != node);
        let placed = Arc::new(Session {
            node,
            class,
            trusted,
            moved_from,
            placed: now,
            requests: AtomicU32::new(0),
            idle_from: AtomicU64::new(0),
        });
        self.untrusted += usize::from(!trusted);
        self.count(node, class, 1);
        if let Some(from) = moved_from {
            self.moved_in[node][from] += 1;
        }
        self.sessions.insert(key.clone(), Arc::clone(&placed));
        Some(placed)
    }

    /// Ends the placement of `key`, if it has one, and returns it.
    pub fn forget(&mut self, key: &Key) -> Option<Arc<Session>> {
        let placed = self.sessions.remove(key)?;
        self.release(&placed);
        Some(placed)
    }

    /// Ends the placements of the sessions that went `idle` without a request, and returns when the next one will
    /// idle out.
    pub fn expire(&mut self, now: Instant, idle: Duration) -> Option<Instant> {
        let idled = |placed: &Session| !placed.busy() && now >= placed.since() + idle;
        let ended: Vec<Key> = self
            .sessions
            .iter()
            .filter(|(_, placed)| idled(placed))
            .map(|(key, _)| key.clone())
            .collect();
        for key in ended {
            self.forget(&key);
        }
        self.sessions
            .values()
            .filter(|placed| !placed.busy())
            .map(|placed| placed.since() + idle)
            .min()
    }

    /// Every placement, for a reload's successor.
    pub fn each(&self) -> impl Iterator<Item = (&Key, &Session)> {
        self.sessions.iter().map(|(key, placed)| (key, &**placed))
    }

    /// The number of sessions placed.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether a new session of a `trusted` tenant or not has room. A full table, or an untrusted session where
    /// untrusted tenants hold their half, forgets the placement idle longest that the new session may take the
    /// place of; there is no room only when each of those is busy.
    fn make_room(&mut self, trusted: bool) -> bool {
        let untrusted_full = !trusted && self.untrusted >= SESSIONS / 2;
        if !untrusted_full && self.sessions.len() < SESSIONS {
            return true;
        }
        let idlest = self
            .sessions
            .iter()
            .filter(|(_, placed)| !placed.busy() && !(untrusted_full && placed.trusted))
            .min_by_key(|(_, placed)| placed.since())
            .map(|(key, _)| key.clone());
        match idlest {
            Some(key) => {
                self.forget(&key);
                true
            }
            None => false,
        }
    }

    /// Frees what a placement that ended held: its node's count, the node it was moved from, and its place in the
    /// untrusted tenants' half.
    fn release(&mut self, placed: &Session) {
        self.untrusted -= usize::from(!placed.trusted);
        self.count(placed.node, placed.class, -1);
        if let Some(from) = placed.moved_from {
            let moved = &mut self.moved_in[placed.node][from];
            *moved = moved.saturating_sub(1);
        }
    }

    fn count(&mut self, node: usize, class: SloClass, change: i32) {
        let count = &mut self.held[node][class as usize];
        *count = count.saturating_add_signed(change);
        self.gauges[node][class as usize].set(i64::from(*count));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(tenant: usize, id: &str) -> Key {
        Key {
            tenant,
            session: SessionId::new(id).unwrap(),
        }
    }

    fn table(nodes: usize) -> Table {
        Table::new(
            (0..nodes)
                .map(|_| std::array::from_fn(|_| Gauge::default()))
                .collect(),
        )
    }

    fn node_of(table: &Table, key: &Key) -> Option<usize> {
        match table.find(key, |_| true) {
            Found::On(placed) => Some(placed.node),
            _ => None,
        }
    }

    #[test]
    fn a_session_keeps_its_node_while_the_node_is_available() {
        let mut table = table(2);
        let now = Instant::now();
        let call = key(0, "call");
        table.place(&call, SloClass::Realtime, 1, true, None, now);
        assert!(matches!(table.find(&call, |_| true), Found::On(p) if p.node == 1));
        assert!(matches!(table.find(&call, |node| node != 1), Found::Lost(p) if p.node == 1));
        // Two trusted tenants may name sessions alike: they are two sessions.
        assert!(matches!(table.find(&key(1, "call"), |_| true), Found::None));
        assert_eq!(table.held(1), [1, 0, 0]);
        assert_eq!(table.gauges[1][0].get(), 1);
    }

    #[test]
    fn a_moved_call_counts_for_its_old_node_until_it_ends() {
        let mut table = table(2);
        let now = Instant::now();
        let call = key(0, "call");
        table.place(&call, SloClass::Realtime, 0, true, None, now);
        table.place(&call, SloClass::Realtime, 1, true, Some(0), now);
        assert_eq!((table.held(0), table.held(1)), ([0, 0, 0], [1, 0, 0]));
        assert_eq!(table.moved_in(1).collect::<Vec<_>>(), [(0, 1)]);
        // An agent's session carries no call for another node, nor does a call placed back where it was.
        table.place(
            &key(0, "agent"),
            SloClass::Background,
            1,
            true,
            Some(0),
            now,
        );
        assert_eq!(table.moved_in(1).collect::<Vec<_>>(), [(0, 1)]);
        table.forget(&call);
        assert_eq!(table.moved_in(1).count(), 0);
        assert_eq!(table.held(1), [0, 0, 1]);
    }

    #[test]
    fn a_session_idles_out_after_idle_without_a_request() {
        let mut table = table(1);
        let idle = Duration::from_secs(60);
        let t0 = Instant::now();
        let agent = key(0, "agent");
        let placed = table
            .place(&agent, SloClass::Background, 0, true, None, t0)
            .unwrap();
        placed.began();
        // Busy, it never idles out.
        assert_eq!(table.expire(t0 + 2 * idle, idle), None);
        assert_eq!(node_of(&table, &agent), Some(0));
        placed.ended(t0 + Duration::from_secs(10));
        assert_eq!(
            table.expire(t0 + Duration::from_secs(20), idle),
            Some(t0 + Duration::from_secs(70))
        );
        assert_eq!(table.expire(t0 + Duration::from_secs(70), idle), None);
        assert_eq!(node_of(&table, &agent), None);
        assert_eq!(table.held(0), [0, 0, 0]);
    }

    #[test]
    fn a_full_table_forgets_the_session_idle_longest_and_never_a_busy_one() {
        let mut table = table(1);
        let t0 = Instant::now();
        let mut busy = Vec::new();
        for i in 0..SESSIONS {
            let at = t0 + Duration::from_millis(i as u64);
            let placed = table
                .place(
                    &key(0, &format!("s{i}")),
                    SloClass::Background,
                    0,
                    true,
                    None,
                    at,
                )
                .unwrap();
            if i == 0 {
                // The oldest is busy, so the next oldest goes.
                placed.began();
                busy.push(placed);
            }
        }
        let later = t0 + Duration::from_secs(60);
        assert!(
            table
                .place(&key(0, "new"), SloClass::Background, 0, true, None, later)
                .is_some()
        );
        assert_eq!(table.len(), SESSIONS);
        assert_eq!(node_of(&table, &key(0, "s0")), Some(0));
        assert_eq!(node_of(&table, &key(0, "s1")), None);
        // With every session busy, a new one finds no room.
        for (key, placed) in table.sessions.iter() {
            if key.session.as_str() != "s0" {
                placed.began();
            }
        }
        assert!(
            table
                .place(&key(0, "newer"), SloClass::Background, 0, true, None, later)
                .is_none()
        );
    }

    #[test]
    fn untrusted_sessions_take_at_most_half_the_table() {
        let mut table = table(1);
        let t0 = Instant::now();
        for i in 0..SESSIONS / 2 {
            let at = t0 + Duration::from_millis(i as u64);
            table.place(
                &key(1, &format!("u{i}")),
                SloClass::Interactive,
                0,
                false,
                None,
                at,
            );
        }
        table.place(&key(0, "trusted"), SloClass::Background, 0, true, None, t0);
        // Another untrusted session takes the place of the untrusted one idle longest, never a trusted one.
        let later = t0 + Duration::from_secs(60);
        table.place(
            &key(1, "more"),
            SloClass::Interactive,
            0,
            false,
            None,
            later,
        );
        assert_eq!(table.untrusted, SESSIONS / 2);
        assert_eq!(node_of(&table, &key(0, "trusted")), Some(0));
        assert_eq!(node_of(&table, &key(1, "u0")), None);
    }
}
