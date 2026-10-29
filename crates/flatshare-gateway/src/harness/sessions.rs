//! The sessions the admission point remembers, in one table by tenant and session id: a session's last usage, from
//! which the tokens of its next background request are estimated, and its turn, where the lifecycle policy follows
//! it ([`Realtime`](super::realtime::Realtime)). Two tenants that name the same session id have a session each.
//!
//! The table holds at most [`SESSIONS`] sessions, whatever clients send. A new session that finds it full takes the
//! place of the session seen longest ago that is not in a turn; only while every session is in a turn is it not
//! remembered. Untrusted tenants' sessions take at most half the table. Each of their requests may name a session of
//! its own, and in a full table a flood of them takes the places of the sessions seen longest ago, trusted ones too,
//! but only until untrusted sessions hold half: from then on they take each other's, so the trusted tenants keep up
//! to half the table.
//!
//! The session seen longest ago is found in an order of the sessions by when they were seen, one for the trusted
//! tenants and one for the others. A session seen again keeps its place there until a new session that needs room
//! comes to it, and the place then moves to when the session was last seen. So seeing a session costs a lookup, as
//! without the orders, and a new session in a full table costs a step in an order for each session in a turn, or
//! seen again, before the one it replaces, where it was a pass over every session.

use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::sync::Arc;
use std::time::Instant;

use flatshare_core::admission::{self, Estimate, SessionUsage};
use flatshare_proto::SessionId;

use super::realtime::Turn;

/// Sessions remembered at most, a bound on what clients can make the gateway hold.
pub(super) const SESSIONS: usize = 4096;

/// A session, by its tenant's name and its id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key {
    pub tenant: Arc<str>,
    pub session: SessionId,
}

/// When a session was seen, and its rank among the sessions seen: of two seen at one instant, the one seen first
/// was seen longer ago.
type Seen = (Instant, u64);

/// What the admission point remembers of a session.
#[derive(Debug)]
pub(super) struct Entry {
    /// Whether its tenant is trusted.
    trusted: bool,
    /// When a request or an event of it last arrived, or its last background request reported its usage.
    seen: Seen,
    /// Its place in its tenants' order ([`Sessions::by_seen`]): when it was seen then, no later than `seen`.
    place: Seen,
    /// The usage its last background request reported.
    pub usage: Option<SessionUsage>,
    /// Its turn, where the lifecycle policy follows it.
    pub turn: Option<Turn>,
}

impl Entry {
    /// Whether the session is in a turn, which keeps it remembered.
    fn busy(&self) -> bool {
        self.turn.as_ref().is_some_and(Turn::busy)
    }
}

#[derive(Debug, Default)]
pub(super) struct Sessions {
    entries: HashMap<Key, Entry>,
    /// Every session by its place, the untrusted tenants' (`[0]`) and the trusted ones' (`[1]`).
    by_seen: [BTreeMap<Seen, Key>; 2],
    /// How many times a session has been seen: the next one's rank.
    seen: u64,
    /// The sessions of untrusted tenants: at most half the table.
    untrusted: usize,
}

impl Sessions {
    /// The estimated prompt of a request of the session `key` whose body is `body_bytes`: from the session's last
    /// usage, if the table remembers one.
    pub fn estimate(&self, key: &Key, body_bytes: u64) -> Estimate {
        let last = self.entries.get(key).and_then(|entry| entry.usage);
        admission::estimate_tokens(body_bytes, last)
    }

    /// The session `key`, of a `trusted` tenant or not, seen `now`, and remembered from now on if it was not and the
    /// table has room for it; `None` if it has none. Also returns the session it took the place of, whose turn its
    /// follower forgets.
    pub fn remember(
        &mut self,
        key: &Key,
        trusted: bool,
        now: Instant,
    ) -> (Option<&mut Entry>, Option<Entry>) {
        let seen = (now, self.seen);
        self.seen += 1;
        let mut forgotten = None;
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.seen = seen;
                // A place is never later than its session was seen; one seen earlier than before moves now.
                if seen < entry.place {
                    let order = &mut self.by_seen[usize::from(entry.trusted)];
                    let key = order.remove(&entry.place).expect("a session has its place");
                    order.insert(seen, key);
                    entry.place = seen;
                }
            }
            None => {
                let Some(room) = self.make_room(trusted) else {
                    return (None, None);
                };
                forgotten = room;
                self.untrusted += usize::from(!trusted);
                self.by_seen[usize::from(trusted)].insert(seen, key.clone());
                let entry = Entry {
                    trusted,
                    seen,
                    place: seen,
                    usage: None,
                    turn: None,
                };
                self.entries.insert(key.clone(), entry);
            }
        }
        (self.entries.get_mut(key), forgotten)
    }

    /// The turn of the session `key`, if the table remembers one.
    pub fn turn_mut(&mut self, key: &Key) -> Option<&mut Turn> {
        self.entries.get_mut(key)?.turn.as_mut()
    }

    /// The sessions' turns.
    pub fn turns(&self) -> impl Iterator<Item = (&Key, &Turn)> {
        self.entries
            .iter()
            .filter_map(|(key, entry)| Some((key, entry.turn.as_ref()?)))
    }

    /// Forgets the turn of the session `key`, and the session with it when the table remembers nothing else of it.
    pub fn end_turn(&mut self, key: &Key) -> Option<Turn> {
        let entry = self.entries.get_mut(key)?;
        let turn = entry.turn.take();
        if entry.usage.is_none() {
            self.forget(key);
        }
        turn
    }

    /// Room for a new session of a `trusted` tenant or not: `Some` with the session whose place it takes, if it
    /// takes one, and `None` when there is none. A full table, or an untrusted session where untrusted tenants hold
    /// their half, forgets the session seen longest ago that is not in a turn and whose place the new one may take:
    /// an untrusted one's then, any one's otherwise. The orders are looked at only when the table, or the untrusted
    /// half, is full.
    fn make_room(&mut self, trusted: bool) -> Option<Option<Entry>> {
        let untrusted_full = !trusted && self.untrusted >= SESSIONS / 2;
        if !untrusted_full && self.entries.len() < SESSIONS {
            return Some(None);
        }
        let untrusted = self.idlest(0);
        let trusted = if untrusted_full { None } else { self.idlest(1) };
        let (order, place) = match (untrusted, trusted) {
            (Some(untrusted), Some(trusted)) if trusted < untrusted => (1, trusted),
            (Some(untrusted), _) => (0, untrusted),
            (None, Some(trusted)) => (1, trusted),
            (None, None) => return None,
        };
        let key = self.by_seen[order]
            .remove(&place)
            .expect("a place has its session");
        let entry = self
            .entries
            .remove(&key)
            .expect("a place's session is in the table");
        self.untrusted -= usize::from(!entry.trusted);
        Some(Some(entry))
    }

    /// The place, in order `order`, of the session seen longest ago that is not in a turn, if one is not. A place it
    /// passes that is earlier than when its session was last seen moves there, and is looked at again in its turn.
    fn idlest(&mut self, order: usize) -> Option<Seen> {
        let mut after = None;
        loop {
            let (place, seen, busy) = {
                let mut places = match after {
                    None => self.by_seen[order].range(..),
                    Some(after) => {
                        self.by_seen[order].range((Bound::Excluded(after), Bound::Unbounded))
                    }
                };
                let (place, key) = places.next()?;
                let entry = &self.entries[key];
                (*place, entry.seen, entry.busy())
            };
            if seen != place {
                let key = self.by_seen[order]
                    .remove(&place)
                    .expect("a place has its session");
                self.entries
                    .get_mut(&key)
                    .expect("a place's session is in the table")
                    .place = seen;
                self.by_seen[order].insert(seen, key);
            } else if busy {
                after = Some(place);
            } else {
                return Some(place);
            }
        }
    }

    fn forget(&mut self, key: &Key) -> Option<Entry> {
        let entry = self.entries.remove(key)?;
        self.by_seen[usize::from(entry.trusted)].remove(&entry.place);
        self.untrusted -= usize::from(!entry.trusted);
        Some(entry)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The tests: the session table's bound, its untrusted half and its tenants.
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn key(tenant: &str, session: &str) -> Key {
        Key {
            tenant: tenant.into(),
            session: SessionId::new(session).unwrap(),
        }
    }

    fn usage(prompt_tokens: u64) -> SessionUsage {
        SessionUsage {
            prompt_tokens,
            body_bytes: 400,
        }
    }

    /// Remembers `key`'s usage at `at` seconds after `t0`, as a background request's end does.
    fn observe(table: &mut Sessions, key: &Key, trusted: bool, prompt_tokens: u64, at: Instant) {
        if let (Some(entry), _) = table.remember(key, trusted, at) {
            entry.usage = Some(usage(prompt_tokens));
        }
    }

    #[test]
    fn the_table_is_bounded_and_forgets_the_session_seen_longest_ago() {
        let (mut table, t0) = (Sessions::default(), Instant::now());
        let at = |n: usize| t0 + Duration::from_millis(n as u64);
        for n in 0..SESSIONS {
            observe(
                &mut table,
                &key("agents", &format!("s-{n}")),
                true,
                n as u64 + 1,
                at(n),
            );
        }
        // s-0 is seen again: s-1 is now the one seen longest ago.
        observe(&mut table, &key("agents", "s-0"), true, 7, at(SESSIONS));
        observe(&mut table, &key("agents", "new"), true, 9, at(SESSIONS + 1));
        assert_eq!(table.len(), SESSIONS);
        let estimate =
            |table: &Sessions, session: &str| table.estimate(&key("agents", session), 400).tokens;
        // Forgotten: a quarter token per byte again.
        assert_eq!(estimate(&table, "s-1"), 100);
        assert_eq!(
            (
                estimate(&table, "s-0"),
                estimate(&table, "s-2"),
                estimate(&table, "new")
            ),
            (7, 3, 9)
        );
    }

    #[test]
    fn sessions_in_a_turn_are_kept_and_a_new_one_finds_no_room_while_all_are() {
        let (mut table, t0) = (Sessions::default(), Instant::now());
        for n in 0..SESSIONS {
            let (entry, _) = table.remember(&key("voice", &format!("voice-{n}")), true, t0);
            entry.unwrap().turn = Some(Turn::in_flight(t0));
        }
        let later = t0 + Duration::from_secs(1);
        let (entry, forgotten) = table.remember(&key("voice", "voice-new"), true, later);
        assert!(entry.is_none() && forgotten.is_none());
        // A session out of its turn may be forgotten for a new one.
        let idle = key("voice", "voice-7");
        table.end_turn(&idle);
        assert_eq!(table.len(), SESSIONS - 1);
        observe(&mut table, &key("agents", "s-1"), true, 5, later);
        let (entry, forgotten) = table.remember(&key("voice", "voice-new"), true, later);
        assert!(entry.is_some());
        assert_eq!(forgotten.map(|entry| entry.usage), Some(Some(usage(5))));
    }

    #[test]
    fn untrusted_tenants_take_at_most_half_the_table() {
        let (mut table, t0) = (Sessions::default(), Instant::now());
        let at = |n: usize| t0 + Duration::from_millis(n as u64);
        // The agents' sessions, seen before any of the flood's.
        for n in 0..SESSIONS / 2 {
            observe(
                &mut table,
                &key("agents", &format!("agent-{n}")),
                true,
                5,
                at(n),
            );
        }
        // A flood of untrusted sessions, each a prompt_cache_key of its own, twice the table.
        for n in 0..2 * SESSIONS {
            observe(
                &mut table,
                &key("bench", &format!("h-{n}")),
                false,
                5,
                at(SESSIONS + n),
            );
        }
        assert_eq!((table.len(), table.untrusted), (SESSIONS, SESSIONS / 2));
        for n in 0..SESSIONS / 2 {
            let agent = key("agents", &format!("agent-{n}"));
            assert_eq!(
                table.estimate(&agent, 400).tokens,
                5,
                "agent-{n} was forgotten"
            );
        }
        // Only the flood's latest half is remembered.
        assert_eq!(table.estimate(&key("bench", "h-0"), 400).tokens, 100);
        let last = key("bench", &format!("h-{}", 2 * SESSIONS - 1));
        assert_eq!(table.estimate(&last, 400).tokens, 5);
        // A trusted session takes the place of the session seen longest ago, the earliest agent's.
        observe(
            &mut table,
            &key("agents", "late"),
            true,
            5,
            at(4 * SESSIONS),
        );
        assert_eq!(table.estimate(&key("agents", "agent-0"), 400).tokens, 100);
        assert_eq!(table.untrusted, SESSIONS / 2);
    }

    #[test]
    fn two_tenants_that_name_one_session_have_a_session_each() {
        let (mut table, t0) = (Sessions::default(), Instant::now());
        observe(&mut table, &key("agents", "s-1"), true, 900, t0);
        assert_eq!(table.estimate(&key("agents", "s-1"), 400).tokens, 900);
        assert_eq!(table.estimate(&key("voice", "s-1"), 400).tokens, 100);
    }

    /// The table as it was before its orders: each session's trust, when it was seen and whether it is in a turn,
    /// its last usage's token count, and a pass over all of them for the session a new one replaces.
    #[derive(Default)]
    struct Scan {
        sessions: HashMap<Key, (bool, Instant, bool, Option<u64>)>,
        untrusted: usize,
        /// The sessions an untrusted one replaced while untrusted tenants held their half.
        in_their_half: usize,
    }

    impl Scan {
        /// Whether the session is remembered, and the tokens of the session whose place it took.
        fn remember(
            &mut self,
            key: &Key,
            trusted: bool,
            now: Instant,
        ) -> (bool, Option<Option<u64>>) {
            let mut forgotten = None;
            if !self.sessions.contains_key(key) {
                let untrusted_full = !trusted && self.untrusted >= SESSIONS / 2;
                if untrusted_full || self.sessions.len() >= SESSIONS {
                    let idlest = self
                        .sessions
                        .iter()
                        .filter(|(_, (trusted, _, busy, _))| !busy && !(untrusted_full && *trusted))
                        .min_by_key(|(_, (_, seen, _, _))| *seen)
                        .map(|(key, _)| key.clone());
                    let Some(idlest) = idlest else {
                        return (false, None);
                    };
                    let (was_trusted, _, _, usage) = self.sessions.remove(&idlest).unwrap();
                    self.untrusted -= usize::from(!was_trusted);
                    self.in_their_half += usize::from(untrusted_full);
                    forgotten = Some(usage);
                }
                self.untrusted += usize::from(!trusted);
                self.sessions
                    .insert(key.clone(), (trusted, now, false, None));
            }
            self.sessions.get_mut(key).unwrap().1 = now;
            (true, forgotten)
        }
    }

    #[test]
    fn a_new_session_takes_the_place_the_pass_over_the_table_gave_it() {
        // The orders replaced a pass over every session: the same session goes, step by step, for sessions seen
        // again, at times that go back as well as forward, in turns and out of them, of trusted and untrusted
        // tenants, in a table full and in one whose untrusted half is.
        let (mut table, mut scan, t0) = (Sessions::default(), Scan::default(), Instant::now());
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        let mut known: Vec<(Key, bool)> = Vec::new();
        let (mut replaced, mut refused) = (0, 0);
        for step in 0..3 * SESSIONS {
            // Times are distinct, as the scan's choice is only then one session; one step in ten goes back.
            let tick = if random(10) == 0 {
                10 * step.saturating_sub(40) + 7
            } else {
                10 * step
            };
            let now = t0 + Duration::from_micros(tick as u64);
            let (key, trusted) = match random(100) {
                0..45 if !known.is_empty() => known[random(known.len())].clone(),
                _ => {
                    let trusted = random(2) > 0;
                    let tenant = if trusted { "agents" } else { "bench" };
                    let fresh = (key(tenant, &format!("s-{step}")), trusted);
                    known.push(fresh.clone());
                    fresh
                }
            };
            let turn = random(10) == 0;
            let tokens = step as u64;
            let (entry, forgotten) = table.remember(&key, trusted, now);
            let (remembered, scanned) = scan.remember(&key, trusted, now);
            assert_eq!(entry.is_some(), remembered, "step {step}");
            assert_eq!(
                forgotten.map(|entry| entry.usage.map(|u| u.prompt_tokens)),
                scanned,
                "step {step}"
            );
            replaced += usize::from(scanned.is_some());
            refused += usize::from(!remembered);
            if let Some(entry) = entry {
                let mirror = scan.sessions.get_mut(&key).unwrap();
                if turn {
                    entry.turn = Some(Turn::in_flight(now));
                    mirror.2 = true;
                } else {
                    entry.usage = Some(usage(tokens));
                    mirror.3 = Some(tokens);
                }
            }
            // A turn ends now and then: its session goes with it if it has no usage.
            if random(8) == 0 {
                let (key, _) = &known[random(known.len())];
                if table.end_turn(key).is_some() {
                    let mirror = scan.sessions.get_mut(key).unwrap();
                    mirror.2 = false;
                    if mirror.3.is_none() {
                        let (trusted, ..) = scan.sessions.remove(key).unwrap();
                        scan.untrusted -= usize::from(!trusted);
                    }
                }
            }
            assert_eq!(
                (table.len(), table.untrusted),
                (scan.sessions.len(), scan.untrusted),
                "step {step}"
            );
        }
        assert!(
            replaced > SESSIONS / 2 && scan.in_their_half > SESSIONS / 8 && refused == 0,
            "{replaced} replaced, {} of them in the untrusted half; {refused} refused",
            scan.in_their_half
        );
    }

    #[test]
    fn of_sessions_seen_at_one_instant_the_one_seen_first_goes_first() {
        // The pass over the table took whichever its hash order met first; the orders take the one seen first.
        let (mut table, t0) = (Sessions::default(), Instant::now());
        for n in 0..SESSIONS {
            observe(
                &mut table,
                &key("agents", &format!("s-{n}")),
                true,
                n as u64,
                t0,
            );
        }
        for n in 0..3 {
            let (_, forgotten) = table.remember(&key("agents", &format!("new-{n}")), true, t0);
            assert_eq!(forgotten.and_then(|entry| entry.usage), Some(usage(n)));
        }
    }

    /// What a session costs the admission point's lock as the table fills: a new session when the table is full
    /// (it takes the place of the session seen longest ago), a new untrusted one while untrusted tenants hold their
    /// half (theirs), and a session the table has (seen again). `cargo test --release -p flatshare-gateway --lib
    /// sessions -- --ignored --nocapture` prints them.
    #[test]
    #[ignore = "a benchmark, for --release"]
    fn what_a_session_costs_as_the_table_fills() {
        const N: usize = 200_000;
        let t0 = Instant::now();
        let at = |n: usize| t0 + Duration::from_micros(n as u64);
        let keys: Vec<Key> = (0..N).map(|n| key("agents", &format!("s-{n}"))).collect();
        let flood: Vec<Key> = (0..N).map(|n| key("bench", &format!("h-{n}"))).collect();
        let per = |run: &mut dyn FnMut()| {
            let start = Instant::now();
            run();
            start.elapsed().as_nanos() as f64 / N as f64
        };
        let full = || {
            let mut table = Sessions::default();
            for (n, key) in keys[..SESSIONS].iter().enumerate() {
                observe(&mut table, key, true, 5, at(n));
            }
            table
        };
        let mut table = full();
        let mut n = SESSIONS;
        let new = per(&mut || {
            for key in &keys[SESSIONS..] {
                n += 1;
                std::hint::black_box(table.remember(key, true, at(n)).1);
            }
        });
        // Half the table the agents', seen first; then the flood fills the other half and goes on.
        let mut table = Sessions::default();
        for (n, key) in keys[..SESSIONS / 2].iter().enumerate() {
            observe(&mut table, key, true, 5, at(n));
        }
        let mut n = SESSIONS / 2;
        let untrusted = per(&mut || {
            for key in &flood {
                n += 1;
                std::hint::black_box(table.remember(key, false, at(n)).1);
            }
        });
        let mut table = full();
        let mut n = SESSIONS;
        let again = per(&mut || {
            for i in 0..N {
                n += 1;
                let key = &keys[i % SESSIONS];
                std::hint::black_box(table.remember(key, true, at(n)).0.is_some());
            }
        });
        // Nine sessions the table has seen again for each new one, which replaces the one seen longest ago.
        let mut table = full();
        let mut n = SESSIONS;
        let mut fresh = keys[SESSIONS..].iter();
        let mixed = per(&mut || {
            for i in 0..N {
                n += 1;
                if i % 10 == 9 {
                    std::hint::black_box(table.remember(fresh.next().unwrap(), true, at(n)).1);
                } else {
                    let key = &keys[(i * 7919) % SESSIONS];
                    std::hint::black_box(table.remember(key, true, at(n)).0.is_some());
                }
            }
        });
        println!(
            "SESSIONS a new session in a full table {new:.0} ns, a new untrusted one in their full half \
             {untrusted:.0} ns, one the table has {again:.0} ns, nine seen again for each new one {mixed:.0} ns"
        );
    }
}
