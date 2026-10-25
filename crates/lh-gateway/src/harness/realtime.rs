//! The lifecycle policy's view of the realtime sessions: where each one is in its turn, from the lifecycle events
//! its emitter posts and from its own realtime requests. `lh_core::admission` decides from how many sessions are
//! in each phase ([`Turns`]); this module keeps the phases, ends those that time out, and counts the headroom's
//! false alarms. A session's turn is kept in the admission point's table of sessions ([`Sessions`]), which
//! remembers a session in a turn for as long as the turn lasts.
//!
//! A session goes `Idle` → `UserSpeaking` (`user_speech_started`) → `Critical` (`user_turn_ended`, or its
//! realtime request if that comes first, as LiveKit's preemptive generation sends it) → `Playback`
//! (`playback_started`, the reply's first audio) → `Idle` (`playback_ended`). A barge-in (`interrupted`) is the
//! user speaking again, and `session_closed` forgets the session's turn.
//! - **Lost or reordered events.** An event moves its session to the event's phase whatever the phase was, in the
//!   order the gateway receives events, so a lost or late event costs the phases until the next one. Every phase
//!   but `Idle` ends after `state_timeout_s` without an event or request, counted.
//! - **No events.** A session that has posted none (its emitter sends none, or events are off) follows its
//!   realtime requests alone: it is `Critical` while one is in flight, and forgotten when the last one ends.
//! - **False alarms.** An arming of the headroom (from speech onset, or only in the critical window) is followed
//!   if a realtime request of the session arrives before the session leaves the armed phases; otherwise it is a
//!   false alarm, charged the token-seconds of background work the headroom held back meanwhile.

use std::time::{Duration, Instant};

use lh_core::admission::{Trigger, Turns};
use lh_proto::EventKind;
use prometheus_client::metrics::gauge::Gauge;

use super::metrics::LifecycleMetrics;
use super::sessions::{Key, Sessions};

/// Where a realtime session is in its turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Idle,
    UserSpeaking,
    /// From the end of the user's turn, or the realtime request if it comes first, to the reply's first audio.
    Critical,
    Playback,
}

impl Phase {
    pub const ALL: [Self; 4] = [
        Self::Idle,
        Self::UserSpeaking,
        Self::Critical,
        Self::Playback,
    ];

    /// The name, also a metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::UserSpeaking => "user_speaking",
            Self::Critical => "critical",
            Self::Playback => "playback",
        }
    }

    /// The counts of one session in this phase.
    const fn turns(self) -> Turns {
        Turns {
            speaking: matches!(self, Self::UserSpeaking) as u32,
            critical: matches!(self, Self::Critical) as u32,
        }
    }
}

/// The realtime sessions' phases, counted.
#[derive(Debug)]
pub(crate) struct Realtime {
    /// Sessions in each phase, in [`Phase::ALL`] order, and their gauges.
    counts: [u32; 4],
    gauges: [Gauge; 4],
    /// What arms the headroom; `None` with the headroom off.
    trigger: Option<Trigger>,
    timeout: Duration,
    held: HeldBack,
    metrics: LifecycleMetrics,
}

/// Where a session is in its turn.
#[derive(Debug)]
pub(crate) struct Turn {
    phase: Phase,
    /// When the session entered its phase, or an event or a request last put it there.
    since: Instant,
    /// Whether the session has posted a lifecycle event; until it does, its realtime requests drive it.
    evented: bool,
    /// Its realtime requests in flight.
    requests: u32,
    /// An arming of the headroom that no realtime request has followed yet: the token-seconds held back when
    /// it began.
    armed: Option<f64>,
}

impl Turn {
    /// Whether the session is in a turn: out of `Idle`, or with a realtime request in flight.
    pub fn busy(&self) -> bool {
        self.phase != Phase::Idle || self.requests > 0
    }

    /// A session with a realtime request in flight since `now`.
    #[cfg(test)]
    pub fn in_flight(now: Instant) -> Self {
        Self {
            phase: Phase::Critical,
            since: now,
            evented: false,
            requests: 1,
            armed: None,
        }
    }
}

/// The integral over time of the estimated background tokens the headroom holds back.
#[derive(Debug)]
struct HeldBack {
    token_seconds: f64,
    tokens: u64,
    at: Instant,
}

impl HeldBack {
    fn until(&self, now: Instant) -> f64 {
        let held = now.saturating_duration_since(self.at).as_secs_f64();
        self.token_seconds + self.tokens as f64 * held
    }
}

impl Realtime {
    pub fn new(trigger: Option<Trigger>, timeout: Duration, metrics: LifecycleMetrics) -> Self {
        Self {
            counts: [0; 4],
            gauges: Phase::ALL.map(|phase| metrics.realtime_sessions(phase)),
            trigger,
            timeout,
            held: HeldBack {
                token_seconds: 0.0,
                tokens: 0,
                at: Instant::now(),
            },
            metrics,
        }
    }

    /// The sessions in each phase the policy acts on.
    pub fn turns(&self) -> Turns {
        Turns {
            speaking: self.counts[Phase::UserSpeaking as usize],
            critical: self.counts[Phase::Critical as usize],
        }
    }

    /// From `now` on, the headroom holds back `tokens` of background work.
    pub fn held_back(&mut self, now: Instant, tokens: u64) {
        self.held.token_seconds = self.held.until(now);
        self.held.tokens = tokens;
        self.held.at = now;
    }

    /// A lifecycle event of the session `key`, of a `trusted` tenant or not, received now.
    pub fn event(
        &mut self,
        sessions: &mut Sessions,
        key: &Key,
        trusted: bool,
        kind: &EventKind,
        now: Instant,
    ) {
        let phase = match kind {
            EventKind::UserSpeechStarted | EventKind::Interrupted { .. } => Phase::UserSpeaking,
            EventKind::UserTurnEnded => Phase::Critical,
            EventKind::PlaybackStarted => Phase::Playback,
            EventKind::PlaybackEnded { .. } => Phase::Idle,
            EventKind::SessionClosed => return self.end(sessions, key, now),
            // The agents' events.
            EventKind::TurnStarted
            | EventKind::TurnFinished { .. }
            | EventKind::ToolWaitStarted { .. }
            | EventKind::ToolWaitEnded { .. } => return,
        };
        let Some(turn) = self.follow(sessions, key, trusted, now) else {
            return;
        };
        turn.evented = true;
        self.enter(sessions, key, phase, now);
    }

    /// A realtime request of the session `key`, of a `trusted` tenant or not, arrived: the session is in its
    /// critical window, and an arming of the headroom is followed.
    pub fn request_started(
        &mut self,
        sessions: &mut Sessions,
        key: &Key,
        trusted: bool,
        now: Instant,
    ) {
        let Some(turn) = self.follow(sessions, key, trusted, now) else {
            return;
        };
        turn.requests += 1;
        self.enter(sessions, key, Phase::Critical, now);
        if let Some(turn) = sessions.turn_mut(key) {
            turn.armed = None;
        }
    }

    /// A realtime request of the session `key` ended. A session without events is forgotten with its last request.
    pub fn request_ended(&mut self, sessions: &mut Sessions, key: &Key, now: Instant) {
        let Some(turn) = sessions.turn_mut(key) else {
            return;
        };
        turn.requests = turn.requests.saturating_sub(1);
        if !turn.evented && turn.requests == 0 {
            self.end(sessions, key, now);
        }
    }

    /// Ends the phases that lasted `state_timeout_s`, and returns when the next one will.
    pub fn expire(&mut self, sessions: &mut Sessions, now: Instant) -> Option<Instant> {
        let timeout = self.timeout;
        let ended: Vec<(Key, Phase)> = sessions
            .turns()
            .filter(|(_, turn)| turn.phase != Phase::Idle && now >= turn.since + timeout)
            .map(|(key, turn)| (key.clone(), turn.phase))
            .collect();
        for (key, phase) in ended {
            self.metrics.state_timeout(phase);
            self.enter(sessions, &key, Phase::Idle, now);
        }
        sessions
            .turns()
            .filter(|(_, turn)| turn.phase != Phase::Idle)
            .map(|(_, turn)| turn.since + timeout)
            .min()
    }

    /// The session's turn, followed from `Idle` if it was not yet; `None` if the table has no room for the session.
    fn follow<'s>(
        &mut self,
        sessions: &'s mut Sessions,
        key: &Key,
        trusted: bool,
        now: Instant,
    ) -> Option<&'s mut Turn> {
        let (entry, forgotten) = sessions.remember(key, trusted, now);
        if let Some(turn) = forgotten.and_then(|entry| entry.turn) {
            self.forgotten(turn, now);
        }
        let entry = entry?;
        if entry.turn.is_none() {
            entry.turn = Some(Turn {
                phase: Phase::Idle,
                since: now,
                evented: false,
                requests: 0,
                armed: None,
            });
            self.count(Phase::Idle, 1);
        }
        entry.turn.as_mut()
    }

    /// Moves a followed session to `phase`, or confirms it there, at `now`. An arming that ends here is a false
    /// alarm.
    fn enter(&mut self, sessions: &mut Sessions, key: &Key, phase: Phase, now: Instant) {
        let Some(turn) = sessions.turn_mut(key) else {
            return;
        };
        let was = std::mem::replace(&mut turn.phase, phase);
        turn.since = now;
        let armed = |phase: Phase| self.trigger.is_some_and(|t| t.armed(phase.turns()) > 0);
        match (armed(was), armed(phase)) {
            (false, true) => turn.armed = Some(self.held.until(now)),
            (true, false) => {
                if let Some(at) = turn.armed.take() {
                    self.metrics.false_alarm(self.held.until(now) - at);
                }
            }
            _ => {}
        }
        if was != phase {
            self.count(was, -1);
            self.count(phase, 1);
        }
    }

    /// Stops following a session.
    fn end(&mut self, sessions: &mut Sessions, key: &Key, now: Instant) {
        if let Some(turn) = sessions.end_turn(key) {
            self.forgotten(turn, now);
        }
    }

    /// A turn the table no longer holds, as the session's turn ended or a new session took its place. An arming
    /// it leaves unfollowed is a false alarm.
    pub fn forgotten(&mut self, turn: Turn, now: Instant) {
        if let Some(at) = turn.armed {
            self.metrics.false_alarm(self.held.until(now) - at);
        }
        self.count(turn.phase, -1);
    }

    fn count(&mut self, phase: Phase, change: i32) {
        let count = &mut self.counts[phase as usize];
        *count = count.saturating_add_signed(change);
        self.gauges[phase as usize].set(i64::from(*count));
    }
}

#[cfg(test)]
mod tests {
    use lh_proto::SessionId;

    use super::super::sessions::SESSIONS;
    use super::*;

    /// The tracker and the table of sessions it keeps the turns in, driven as the admission point drives them, for
    /// a trusted tenant's sessions.
    struct Tracker {
        rt: Realtime,
        table: Sessions,
    }

    impl Tracker {
        fn event(&mut self, session: &Key, kind: &EventKind, now: Instant) {
            self.rt.event(&mut self.table, session, true, kind, now);
        }

        fn request_started(&mut self, session: &Key, now: Instant) {
            self.rt.request_started(&mut self.table, session, true, now);
        }

        fn request_ended(&mut self, session: &Key, now: Instant) {
            self.rt.request_ended(&mut self.table, session, now);
        }

        fn expire(&mut self, now: Instant) -> Option<Instant> {
            self.rt.expire(&mut self.table, now)
        }

        fn held_back(&mut self, now: Instant, tokens: u64) {
            self.rt.held_back(now, tokens);
        }

        fn turns(&self) -> Turns {
            self.rt.turns()
        }

        fn counts(&self) -> [u32; 4] {
            self.rt.counts
        }
    }

    fn session(id: &str) -> Key {
        Key {
            tenant: "voice".into(),
            session: SessionId::new(id).unwrap(),
        }
    }

    fn tracker(trigger: Option<Trigger>) -> (Tracker, LifecycleMetrics) {
        let metrics = LifecycleMetrics::new();
        let rt = Realtime::new(trigger, Duration::from_secs(10), metrics.clone());
        let tracker = Tracker {
            rt,
            table: Sessions::default(),
        };
        (tracker, metrics)
    }

    fn turns(speaking: u32, critical: u32) -> Turns {
        Turns { speaking, critical }
    }

    #[test]
    fn a_turn_goes_through_its_phases_and_back_to_idle() {
        let (mut rt, metrics) = tracker(Some(Trigger::Onset));
        let (s, t0) = (session("voice-RM_1"), Instant::now());
        rt.event(&s, &EventKind::UserSpeechStarted, t0);
        assert_eq!(rt.turns(), turns(1, 0));
        rt.event(&s, &EventKind::UserTurnEnded, t0);
        assert_eq!(rt.turns(), turns(0, 1));
        rt.request_started(&s, t0);
        rt.request_ended(&s, t0);
        assert_eq!(
            rt.turns(),
            turns(0, 1),
            "a session with events leaves its window at its first audio"
        );
        rt.event(&s, &EventKind::PlaybackStarted, t0);
        assert_eq!(rt.turns(), turns(0, 0));
        rt.event(&s, &EventKind::Interrupted { played_ms: 900 }, t0);
        assert_eq!(rt.turns(), turns(1, 0), "a barge-in is the user speaking");
        rt.event(&s, &EventKind::SessionClosed, t0);
        assert_eq!((rt.turns(), rt.table.len()), (turns(0, 0), 0));
        // The speaking user's arming, which no request followed, ended with the session.
        assert_eq!(metrics.false_alarms().0, 1);
    }

    #[test]
    fn a_session_without_events_follows_its_requests() {
        let (mut rt, metrics) = tracker(Some(Trigger::Onset));
        let (s, t0) = (session("bench-1"), Instant::now());
        rt.request_started(&s, t0);
        rt.request_started(&s, t0);
        assert_eq!(rt.turns(), turns(0, 1));
        rt.request_ended(&s, t0);
        assert_eq!(rt.turns(), turns(0, 1), "one request is still in flight");
        rt.request_ended(&s, t0);
        assert_eq!((rt.turns(), rt.table.len()), (turns(0, 0), 0));
        assert_eq!(metrics.false_alarms().0, 0);
    }

    #[test]
    fn every_phase_but_idle_times_out() {
        let (mut rt, metrics) = tracker(Some(Trigger::Onset));
        let t0 = Instant::now();
        let (a, b) = (session("voice-a"), session("voice-b"));
        rt.event(&a, &EventKind::UserTurnEnded, t0);
        rt.event(&b, &EventKind::PlaybackStarted, t0 + Duration::from_secs(4));
        assert_eq!(rt.expire(t0), Some(t0 + Duration::from_secs(10)));
        // A lost playback_started: the window ends at the timeout, counted, and so does its arming.
        let next = rt.expire(t0 + Duration::from_secs(10));
        assert_eq!(next, Some(t0 + Duration::from_secs(14)));
        assert_eq!(rt.turns(), turns(0, 0));
        assert_eq!(metrics.false_alarms().0, 1);
        assert_eq!(rt.expire(t0 + Duration::from_secs(14)), None);
        let timeouts = |phase| metrics.state_timeouts(phase);
        assert_eq!(
            (timeouts(Phase::Critical), timeouts(Phase::Playback)),
            (1, 1)
        );
        // An event confirms a phase, and its timeout starts again.
        rt.event(
            &a,
            &EventKind::UserSpeechStarted,
            t0 + Duration::from_secs(20),
        );
        rt.event(
            &a,
            &EventKind::UserSpeechStarted,
            t0 + Duration::from_secs(25),
        );
        assert_eq!(
            rt.expire(t0 + Duration::from_secs(30)),
            Some(t0 + Duration::from_secs(35))
        );
    }

    #[test]
    fn events_move_a_session_whatever_its_phase() {
        let (mut rt, _) = tracker(Some(Trigger::Onset));
        let (s, t0) = (session("voice-RM_2"), Instant::now());
        // The onset was lost: the end of the turn still opens the window, and a late playback_ended ends it.
        rt.event(&s, &EventKind::UserTurnEnded, t0);
        assert_eq!(rt.turns(), turns(0, 1));
        rt.event(&s, &EventKind::PlaybackEnded { played_ms: 2_000 }, t0);
        assert_eq!(rt.turns(), turns(0, 0));
        // The agents' events change nothing.
        rt.event(&s, &EventKind::TurnStarted, t0);
        assert_eq!((rt.turns(), rt.counts()), (turns(0, 0), [1, 0, 0, 0]));
    }

    #[test]
    fn a_false_alarm_is_charged_what_the_headroom_held_back_meanwhile() {
        let (mut rt, metrics) = tracker(Some(Trigger::Onset));
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        rt.held_back(t0, 0);
        let (noise, turn) = (session("voice-noise"), session("voice-turn"));
        rt.event(&noise, &EventKind::UserSpeechStarted, at(1));
        rt.held_back(at(2), 1_000);
        // A real turn armed meanwhile is followed by its request.
        rt.event(&turn, &EventKind::UserSpeechStarted, at(2));
        rt.request_started(&turn, at(3));
        rt.held_back(at(4), 0);
        // The noise's arming times out: 1,000 tokens held back for 2 s.
        rt.expire(at(11));
        let (alarms, token_seconds) = metrics.false_alarms();
        assert_eq!(alarms, 1);
        assert!((token_seconds - 2_000.0).abs() < 1e-6, "{token_seconds}");
    }

    #[test]
    fn the_request_trigger_arms_only_the_critical_window() {
        let (mut rt, metrics) = tracker(Some(Trigger::Request));
        let (s, t0) = (session("voice-RM_3"), Instant::now());
        rt.event(&s, &EventKind::UserSpeechStarted, t0);
        rt.event(&s, &EventKind::PlaybackStarted, t0);
        assert_eq!(metrics.false_alarms().0, 0, "speech alone arms nothing");
        // An end of turn that no request follows is one.
        rt.event(&s, &EventKind::UserTurnEnded, t0);
        rt.event(&s, &EventKind::UserSpeechStarted, t0);
        assert_eq!(metrics.false_alarms().0, 1);
        // Without the headroom nothing is armed.
        let (mut rt, metrics) = tracker(None);
        rt.event(&s, &EventKind::UserTurnEnded, t0);
        rt.event(&s, &EventKind::SessionClosed, t0);
        assert_eq!(metrics.false_alarms().0, 0);
    }

    #[test]
    fn a_new_session_takes_the_place_of_the_idle_one_seen_longest_ago() {
        let (mut rt, _) = tracker(Some(Trigger::Onset));
        let t0 = Instant::now();
        let at = |n: usize| t0 + Duration::from_millis(n as u64);
        rt.event(
            &session("voice-speaking"),
            &EventKind::UserSpeechStarted,
            t0,
        );
        for n in 1..SESSIONS {
            let s = session(&format!("voice-{n}"));
            rt.event(&s, &EventKind::PlaybackEnded { played_ms: 0 }, at(n));
        }
        assert_eq!(rt.table.len(), SESSIONS);
        rt.event(
            &session("voice-new"),
            &EventKind::UserTurnEnded,
            at(SESSIONS),
        );
        // The speaking session, seen first, is in its turn: voice-1 is the idle one seen longest ago.
        assert_eq!(rt.table.len(), SESSIONS);
        assert!(rt.table.turn_mut(&session("voice-1")).is_none());
        assert!(rt.table.turn_mut(&session("voice-2")).is_some());
        let idle = SESSIONS as u32 - 2;
        assert_eq!((rt.turns(), rt.counts()), (turns(1, 1), [idle, 1, 1, 0]));
    }
}
