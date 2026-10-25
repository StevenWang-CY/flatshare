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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use lh_core::admission::{self, Estimate, SessionUsage};
use lh_proto::SessionId;

use super::realtime::Turn;

/// Sessions remembered at most, a bound on what clients can make the gateway hold.
pub(super) const SESSIONS: usize = 4096;

/// A session, by its tenant's name and its id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key {
    pub tenant: Arc<str>,
    pub session: SessionId,
}

/// What the admission point remembers of a session.
#[derive(Debug)]
pub(super) struct Entry {
    /// Whether its tenant is trusted.
    trusted: bool,
    /// When a request or an event of it last arrived, or its last background request reported its usage.
    seen: Instant,
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
        let mut forgotten = None;
        if !self.entries.contains_key(key) {
            let Some(room) = self.make_room(trusted) else {
                return (None, None);
            };
            forgotten = room;
            self.untrusted += usize::from(!trusted);
            let entry = Entry {
                trusted,
                seen: now,
                usage: None,
                turn: None,
            };
            self.entries.insert(key.clone(), entry);
        }
        let mut entry = self.entries.get_mut(key);
        if let Some(entry) = entry.as_deref_mut() {
            entry.seen = now;
        }
        (entry, forgotten)
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
    /// their half, forgets the session seen longest ago that is not in a turn and whose place the new one may take.
    /// The table is scanned only when it, or the untrusted half, is full.
    fn make_room(&mut self, trusted: bool) -> Option<Option<Entry>> {
        let untrusted_full = !trusted && self.untrusted >= SESSIONS / 2;
        if !untrusted_full && self.entries.len() < SESSIONS {
            return Some(None);
        }
        let idlest = self
            .entries
            .iter()
            .filter(|(_, entry)| !entry.busy() && !(untrusted_full && entry.trusted))
            .min_by_key(|(_, entry)| entry.seen)
            .map(|(key, _)| key.clone())?;
        Some(self.forget(&idlest))
    }

    fn forget(&mut self, key: &Key) -> Option<Entry> {
        let entry = self.entries.remove(key)?;
        self.untrusted -= usize::from(!entry.trusted);
        Some(entry)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

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
}
