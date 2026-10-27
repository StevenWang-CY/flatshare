//! The admission point, between resolving a request and sending it upstream. `flatshare_core::admission` decides
//! whether a request goes upstream now, waits or is refused, and which admitted requests give way to it; this
//! module carries the decisions out. Each held request has a watch channel on which it learns what was decided
//! for it, so waiting never polls, and a wait ends after `queue_timeout_s` with a refusal, or under the lifecycle
//! policy after `max_defer_s` with an admission regardless.
//!
//! A held request owns a [`Ticket`] from its arrival to the end of its response, and dropping the ticket is
//! leaving: a client that goes away while it waits, or before its response ends, frees its place at once.
//!
//! What the admission point remembers of a session, by tenant and session id, is in one bounded table
//! ([`Sessions`]): the usage its last background request reported, from which the next one's tokens are estimated,
//! and its turn.
//!
//! Under the lifecycle policy the admission point also follows the realtime sessions' turns ([`Realtime`]), from
//! lifecycle events and realtime requests, and tells the core how many sessions are in each phase whenever that
//! changes. A task ends the phases that time out.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use flatshare_core::admission::{
    self, Action, Deferral, Estimate, Policy, RequestId, SessionUsage,
};
use flatshare_proto::{LifecycleEvent, SessionId, SloClass};
use tokio::sync::watch;

use super::config;
use super::metrics::{LifecycleMetrics, Metrics, Refusal};
use super::realtime::Realtime;
use super::sessions::{Key, Sessions};
use crate::auth::Tenant;

/// The gateway's admission point.
#[derive(Debug)]
pub(crate) struct Admission {
    state: Mutex<State>,
    policy: Policy,
    /// How long a request may wait.
    max_wait: Duration,
    metrics: Metrics,
}

#[derive(Debug)]
struct State {
    core: admission::Admission,
    /// Where each held request learns what was decided for it.
    signals: HashMap<RequestId, watch::Sender<Signal>>,
    /// The sessions the admission point remembers: their usage, and under the lifecycle policy their turns.
    sessions: Sessions,
    /// The lifecycle policy's realtime sessions.
    realtime: Option<Realtime>,
}

/// What a held request was last told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    Waiting,
    Admitted,
    /// Shed to make room: the request is no longer held, and its response must fail.
    Shed,
}

impl Admission {
    /// The admission point and its metric families, the lifecycle policy's only under it; under that policy, with
    /// the task that ends the realtime sessions' phases as they time out. A configuration that the core refuses is
    /// an error that names the key it lacks ([`config::Admission::core`]).
    pub fn new(config: &config::Admission) -> Result<Arc<Self>, String> {
        let core = config.core()?;
        let (metrics, realtime) = match config.policy {
            Policy::Lifecycle(lifecycle) => {
                let metrics = LifecycleMetrics::new();
                let realtime = config
                    .state_timeout
                    .map(|timeout| Realtime::new(lifecycle.headroom, timeout, metrics.clone()));
                (Metrics::new(Some(metrics)), realtime)
            }
            _ => (Metrics::new(None), None),
        };
        let state = State {
            core,
            signals: HashMap::new(),
            sessions: Sessions::default(),
            realtime,
        };
        let admission = Arc::new(Self {
            state: Mutex::new(state),
            policy: config.policy,
            max_wait: config.max_wait,
            metrics,
        });
        if let Some(timeout) = config.state_timeout {
            tokio::spawn(expire(Arc::downgrade(&admission), timeout));
        }
        Ok(admission)
    }

    /// Takes a request of `tenant` and `class` through the admission point. Returns its ticket once it is
    /// admitted, `None` for a class the policy does not hold (it goes upstream at once), or why it was refused.
    pub async fn admit(
        self: &Arc<Self>,
        tenant: &Tenant,
        class: SloClass,
        session: &SessionId,
        body_bytes: usize,
    ) -> Result<Option<Ticket>, Refusal> {
        if !self.policy.holds(class) {
            self.metrics.admitted(class, Duration::ZERO);
            return Ok(None);
        }
        // Background requests are priced where a policy counts their tokens: token-reservation admits by them,
        // preempt-resubmit counts those its preemptions discard, and the lifecycle policy's headroom, throttle and
        // degrade read them.
        let priced = matches!(
            self.policy,
            Policy::TokenReservation { .. } | Policy::PreemptResubmit | Policy::Lifecycle(_)
        );
        let estimated = (priced && class == SloClass::Background).then_some(body_bytes as u64);
        // The lifecycle policy follows a realtime request's session into its critical window.
        let turn = class == SloClass::Realtime && self.lifecycle();
        let key = (estimated.is_some() || turn).then(|| Key {
            tenant: Arc::clone(&tenant.name),
            session: session.clone(),
        });
        // Every held request takes the lock, so what can be done before it is. Most requests are admitted at
        // once, so only one that waits has its signal changed under the lock.
        let (sender, signal) = watch::channel(Signal::Admitted);
        let (id, tokens, deferred, reply_cap) = {
            let mut state = self.lock();
            let estimate = match (&key, estimated) {
                (Some(key), Some(body_bytes)) => state.sessions.estimate(key, body_bytes),
                _ => Estimate::default(),
            };
            let Ok(arrival) = state.core.arrive(class, estimate) else {
                drop(state);
                self.metrics.refused(class, Refusal::QueueFull);
                return Err(Refusal::QueueFull);
            };
            if !arrival.admitted {
                sender.send_replace(Signal::Waiting);
            }
            state.signals.insert(arrival.id, sender);
            state.apply(arrival.actions);
            match &key {
                Some(key) if turn => state.request_started(key, tenant.trusted, Instant::now()),
                _ => state.settle(),
            }
            (
                arrival.id,
                estimate.tokens,
                arrival.deferred,
                arrival.reply_cap,
            )
        };
        if let Some(reason) = deferred {
            self.metrics.deferred(reason);
        }
        if reply_cap.is_some() {
            self.metrics.degraded();
        }
        let mut ticket = Ticket {
            admission: Arc::clone(self),
            id,
            class,
            signal,
            tokens,
            session: key,
            trusted: tenant.trusted,
            estimated,
            turn,
            deferred,
            aged: false,
            reply_cap,
            waited: Duration::ZERO,
            prompt_tokens: None,
            first_byte: false,
            left: false,
        };
        let queued = Instant::now();
        ticket.wait().await?;
        ticket.waited = queued.elapsed();
        self.metrics.admitted(class, ticket.waited);
        if let Some(reason) = deferred {
            self.metrics.deferral(reason, ticket.waited);
        }
        Ok(Some(ticket))
    }

    /// Lifecycle events of `tenant`'s sessions that the gateway accepted, in the order it received them: the
    /// lifecycle policy follows the realtime sessions' turns by them.
    pub fn events<'a>(
        &self,
        tenant: &Tenant,
        events: impl IntoIterator<Item = &'a LifecycleEvent>,
    ) {
        if !self.lifecycle() {
            return;
        }
        self.lock().events(tenant, events, Instant::now());
    }

    /// The policy; `Debug` shows its parameters.
    pub fn policy(&self) -> Policy {
        self.policy
    }

    /// The admission point's metric families, for the gateway's registry.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn lifecycle(&self) -> bool {
        matches!(self.policy, Policy::Lifecycle(_))
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Ends the realtime sessions' phases as they time out, until the admission point is gone. Every phase times
/// out `timeout` after it began, so no phase entered while this sleeps can end before it wakes.
async fn expire(admission: Weak<Admission>, timeout: Duration) {
    let mut next = Instant::now() + timeout;
    loop {
        tokio::time::sleep_until(next.into()).await;
        let Some(admission) = admission.upgrade() else {
            return;
        };
        let now = Instant::now();
        next = admission.lock().expire(now).unwrap_or(now + timeout);
    }
}

impl State {
    /// Tells each request what the core decided for it.
    fn apply(&mut self, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::Admit(id) => self.signal(id, Signal::Admitted),
                Action::Preempt(id) => self.signal(id, Signal::Waiting),
                Action::Shed(id) => {
                    if let Some(signal) = self.signals.remove(&id) {
                        signal.send_replace(Signal::Shed);
                    }
                }
            }
        }
    }

    fn signal(&self, id: RequestId, signal: Signal) {
        if let Some(sender) = self.signals.get(&id) {
            sender.send_replace(signal);
        }
    }

    /// A request leaves the admission point, which may admit others. Returns the request's signal, for the
    /// caller to drop once it has let go of the lock.
    fn leave(&mut self, id: RequestId) -> Option<watch::Sender<Signal>> {
        let actions = self.core.leave(id);
        self.apply(actions);
        self.signals.remove(&id)
    }

    /// A background request of the session `key`, of a `trusted` tenant or not, reported `usage`: the session's next
    /// estimate starts from it.
    fn observe(&mut self, key: &Key, trusted: bool, usage: SessionUsage, now: Instant) {
        let (entry, forgotten) = self.sessions.remember(key, trusted, now);
        if let Some(entry) = entry {
            entry.usage = Some(usage);
        }
        // The session whose place it took may have been followed out of its turn.
        if let (Some(turn), Some(realtime)) =
            (forgotten.and_then(|entry| entry.turn), &mut self.realtime)
        {
            realtime.forgotten(turn, now);
        }
    }

    /// The lifecycle policy: a realtime request of the session `key`, of a `trusted` tenant or not, arrived.
    fn request_started(&mut self, key: &Key, trusted: bool, now: Instant) {
        match &mut self.realtime {
            Some(realtime) => {
                realtime.request_started(&mut self.sessions, key, trusted, now);
                self.turns_changed();
            }
            None => self.settle(),
        }
    }

    /// The lifecycle policy: a realtime request of the session `key` ended.
    fn request_ended(&mut self, key: &Key, now: Instant) {
        match &mut self.realtime {
            Some(realtime) => {
                realtime.request_ended(&mut self.sessions, key, now);
                self.turns_changed();
            }
            None => self.settle(),
        }
    }

    /// The lifecycle policy: lifecycle events of `tenant`'s sessions.
    fn events<'a>(
        &mut self,
        tenant: &Tenant,
        events: impl IntoIterator<Item = &'a LifecycleEvent>,
        now: Instant,
    ) {
        let Some(realtime) = &mut self.realtime else {
            return;
        };
        for event in events {
            let key = Key {
                tenant: Arc::clone(&tenant.name),
                session: event.session_id.clone(),
            };
            realtime.event(&mut self.sessions, &key, tenant.trusted, &event.kind, now);
        }
        self.turns_changed();
    }

    /// The lifecycle policy: ends the realtime sessions' phases that timed out, and returns when the next one will.
    fn expire(&mut self, now: Instant) -> Option<Instant> {
        let soonest = self
            .realtime
            .as_mut()
            .and_then(|realtime| realtime.expire(&mut self.sessions, now));
        self.turns_changed();
        soonest
    }

    /// The lifecycle policy: tells the core how many realtime sessions are in each phase, and carries out what that
    /// frees.
    fn turns_changed(&mut self) {
        if let Some(realtime) = &self.realtime {
            let actions = self.core.set_turns(realtime.turns());
            self.apply(actions);
            self.settle();
        }
    }

    /// The lifecycle policy: notes what the headroom holds back from now on, after a decision that may change it.
    fn settle(&mut self) {
        if let Some(realtime) = &mut self.realtime {
            realtime.held_back(Instant::now(), self.core.held_back());
        }
    }
}

/// A request's place at the admission point, from its arrival to the end of its response.
#[derive(Debug)]
pub(crate) struct Ticket {
    admission: Arc<Admission>,
    id: RequestId,
    class: SloClass,
    signal: watch::Receiver<Signal>,
    /// The request's estimated prompt tokens; 0 where its policy does not price it.
    tokens: u64,
    /// The request's session, where the admission point remembers its usage or its turn.
    session: Option<Key>,
    /// Whether the request's tenant is trusted.
    trusted: bool,
    /// The body size of a request whose tokens were estimated: its usage informs the next estimate for the
    /// session.
    estimated: Option<u64>,
    /// Whether it is a realtime request under the lifecycle policy, whose end may close its session's critical
    /// window.
    turn: bool,
    /// Why the lifecycle policy deferred the request.
    deferred: Option<Deferral>,
    /// Whether it was admitted regardless, after `max_defer_s`.
    aged: bool,
    /// The reply cap of a realtime request the lifecycle policy degraded.
    reply_cap: Option<u32>,
    /// How long the request waited to be admitted.
    waited: Duration,
    /// The prompt tokens the response reported.
    pub prompt_tokens: Option<u64>,
    /// Whether the response's first byte has been recorded.
    first_byte: bool,
    /// Set when the request has already left, so dropping the ticket does nothing.
    left: bool,
}

impl Ticket {
    /// How long the request waited to be admitted, after the decision that made it wait.
    pub fn waited(&self) -> Duration {
        self.waited
    }

    /// Why the lifecycle policy deferred the request, if it did.
    pub fn deferred(&self) -> Option<Deferral> {
        self.deferred
    }

    /// Whether the request was admitted regardless, after waiting `max_defer_s`.
    pub fn aged(&self) -> bool {
        self.aged
    }

    /// The reply cap of a realtime request the lifecycle policy degraded.
    pub fn reply_cap(&self) -> Option<u32> {
        self.reply_cap
    }

    /// Whether the policy may shed the request after admitting it: flow control sheds background requests.
    pub fn sheddable(&self) -> bool {
        self.class == SloClass::Background
            && matches!(self.admission.policy, Policy::FlowControl { .. })
    }

    /// Completes when the request is shed.
    pub async fn shed(&mut self) {
        // The sender goes only after it sent Shed, or when the request leaves, which an admitted request does
        // by dropping this ticket.
        let _ = self.signal.wait_for(|signal| *signal == Signal::Shed).await;
    }

    /// Whether the policy may preempt the request before its first byte: preempt-resubmit preempts
    /// background requests.
    pub fn preemptible(&self) -> bool {
        self.class == SloClass::Background && self.admission.policy == Policy::PreemptResubmit
    }

    /// Completes when the request is preempted: it waits again, and its upstream attempt is to be dropped.
    pub async fn preempted(&mut self) {
        let _ = self
            .signal
            .wait_for(|signal| *signal != Signal::Admitted)
            .await;
    }

    /// Counts the upstream attempt a preemption dropped, and waits until the request is admitted again, for at
    /// most `queue_timeout_s`.
    pub async fn readmitted(&mut self) -> Result<(), Refusal> {
        self.admission.metrics.preempted(self.class, self.tokens);
        self.wait().await?;
        self.admission.metrics.resubmitted(self.class);
        Ok(())
    }

    /// Records the response's first byte, which under preempt-resubmit ends a realtime request's hold on
    /// background work and makes a background request safe from preemption. False if the request was
    /// preempted first: the byte belongs to a dropped attempt and must not be relayed. Only a preemptible
    /// request can be.
    pub fn first_byte(&mut self) -> bool {
        if self.first_byte || self.admission.policy != Policy::PreemptResubmit {
            return true;
        }
        let mut state = self.admission.lock();
        let Some(actions) = state.core.first_byte(self.id) else {
            return false;
        };
        state.apply(actions);
        self.first_byte = true;
        true
    }

    /// Waits until the request is admitted, for at most `queue_timeout_s`; under the lifecycle policy, until it is
    /// admitted regardless after `max_defer_s`.
    async fn wait(&mut self) -> Result<(), Refusal> {
        if *self.signal.borrow() == Signal::Admitted {
            return Ok(());
        }
        let _waiting = self.admission.metrics.waiting(self.class);
        let admitted = self.signal.wait_for(|signal| *signal == Signal::Admitted);
        if let Ok(Ok(_)) = tokio::time::timeout(self.admission.max_wait, admitted).await {
            return Ok(());
        }
        let mut state = self.admission.lock();
        // The timer and the admission can cross: a request admitted meanwhile goes on.
        if !state.core.is_waiting(self.id) {
            return Ok(());
        }
        if self.admission.lifecycle() {
            // The lifecycle policy refuses no request for its wait, so that background work is never starved.
            let oversize = state.core.age(self.id).unwrap_or_default();
            state.signal(self.id, Signal::Admitted);
            state.settle();
            drop(state);
            self.aged = true;
            self.admission.metrics.aged(oversize);
            return Ok(());
        }
        let signal = state.leave(self.id);
        drop(state);
        drop(signal);
        self.left = true;
        self.admission.metrics.refused(self.class, Refusal::Timeout);
        Err(Refusal::Timeout)
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if self.left {
            return;
        }
        let mut state = self.admission.lock();
        let now = Instant::now();
        if let (Some(key), Some(body_bytes), Some(prompt_tokens)) =
            (&self.session, self.estimated, self.prompt_tokens)
        {
            let usage = SessionUsage {
                prompt_tokens,
                body_bytes,
            };
            state.observe(key, self.trusted, usage, now);
        }
        let signal = state.leave(self.id);
        match &self.session {
            Some(key) if self.turn => state.request_ended(key, now),
            _ => state.settle(),
        }
        drop(state);
        drop(signal);
    }
}
