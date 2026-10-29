//! Several GPU nodes behind one gateway (`[[nodes]]`): each node's engine and speech servers, its health as a whole
//! pipeline, the sessions placed on it, and failover.
//!
//! - **Health.** A node is judged every `health_interval_s`, and at once when its engine's state or readiness
//!   changes: its engine must be serving (its own idle-only canary, [`crate::canary`]) and ready (no request has found
//!   it broken without its `/health` answering since, [`crate::upstream::recover`]), and its STT and TTS servers must
//!   answer `/health` within `health_timeout_s`. A node starts unknown and is healthy after its first passed check;
//!   `unhealthy_after` failed checks in a row make it unhealthy, and `healthy_after` passed ones healthy again. A node
//!   is available while it is healthy and its engine ready: one whose engine a request found broken is passed over at
//!   once, before it has failed enough checks to be unhealthy.
//! - **Sticky placement.** A session is placed by its first request, or by `POST /v1/sessions`, on an available node
//!   with its class's role (realtime for realtime sessions, background for interactive and background ones), and stays
//!   there until it has gone `idle_s` without a request, or its tenant ends it, by `DELETE /v1/sessions/{id}` or by
//!   a `session_closed` lifecycle event, as at a call's end ([`placement`]). A
//!   realtime session takes one of its node's `max_calls` call slots, on the node with the fewest calls, or with
//!   `calls = "fill"` on the first in the configuration's order; an agent's takes one of its node's `max_agents` agent
//!   slots, where the node has that bound, on the node with the fewest agents' sessions, then the fewest requests in
//!   flight, so that its prefix cache stays on one engine. An agent's request that names no session takes no slot,
//!   so a node with that bound takes it only while the node holds no call, and one without (`"unbounded"`, or the
//!   background role alone) beside its calls too.
//! - **Failover.** A node that is not available takes no new session, and a session placed on it is placed again by
//!   its next request (or its next `POST /v1/sessions`), a move that is counted. A realtime session that finds no
//!   available realtime node with a free call slot gets 503 with `Retry-After`, and so does an agent's that finds no
//!   available background node with a free agent slot. A node that carries realtime sessions moved to it from another
//!   node covers: its interactive and background requests in flight are held to `failover_background_max_in_flight`
//!   (0 by default, all of them deferred) until the moved sessions end, whether or not the node they came from is
//!   available again, so that a recovery does not leave them beside the node's whole agent load. With
//!   `calls = "fill"` a realtime node after the first covers too while it carries any call, moved or new: it is a warm
//!   spare that took calls only because the nodes before it were full or not available. The first realtime node never
//!   covers for its own calls; its agents are bounded by `max_agents` alone.
//! - **Interrupted turns.** A realtime request whose node fails before its first content (no connection, one that
//!   closed before the response, or a body that broke off before its first frame) is sent once more on another
//!   available node, within the deadlines of its receipt, after the first attempt is dropped; once its first content
//!   has reached the client it is never sent again ([`crate::proxy`]).

pub(crate) mod metrics;
mod placement;
pub(crate) mod routes;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::time::{Duration, Instant};

use flatshare_proto::{EventKind, LifecycleEvent, SessionId, SloClass};
use prometheus_client::metrics::gauge::Gauge;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, watch};
use tokio::time::MissedTickBehavior;

use crate::auth::{Tenant, Tenants};
use crate::canary::{self, Health, Watched};
use crate::config::{CallPlacement, Canary, NodeConfig, Placement, PrioritySignals, UpstreamAddr};
use crate::deadline::Timeout;
use crate::engine::Engine;
use crate::metrics::{EngineState, GaugeGuard};
use crate::upstream::Upstream;

pub(crate) use metrics::Unplaced;
use metrics::{Reason, Series};
use placement::{Found, Session, Table};
pub(crate) use placement::{Key, SESSIONS};

/// A node's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Verdict {
    /// No check has passed yet, nor have `unhealthy_after` failed.
    Unknown,
    Healthy,
    Unhealthy,
}

impl Verdict {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Healthy => "healthy",
            Self::Unhealthy => "unhealthy",
        }
    }
}

/// A node's checks so far.
#[derive(Debug)]
struct Judged {
    verdict: Verdict,
    /// The failed and passed checks in a row.
    failures: u32,
    passes: u32,
    /// Each component's last check: engine, STT, TTS.
    last: [bool; 3],
}

impl Judged {
    /// Records a check that `passed`, with `unhealthy_after` and `healthy_after`.
    fn record(&mut self, passed: bool, unhealthy_after: u32, healthy_after: u32) {
        if passed {
            self.failures = 0;
            self.passes = self.passes.saturating_add(1);
        } else {
            self.passes = 0;
            self.failures = self.failures.saturating_add(1);
        }
        self.verdict = match self.verdict {
            Verdict::Unknown if passed => Verdict::Healthy,
            Verdict::Unknown | Verdict::Healthy if self.failures >= unhealthy_after => {
                Verdict::Unhealthy
            }
            Verdict::Unhealthy if self.passes >= healthy_after => Verdict::Healthy,
            verdict => verdict,
        };
    }
}

/// A speech server of a node, whose `/health` the node's health needs.
#[derive(Debug)]
pub(crate) struct Speech {
    /// As configured, which `POST /v1/sessions` answers.
    pub addr: UpstreamAddr,
    client: Upstream,
}

/// One GPU node.
#[derive(Debug)]
pub(crate) struct Node {
    pub index: usize,
    pub name: Arc<str>,
    pub realtime: bool,
    pub background: bool,
    pub max_calls: usize,
    /// The agents' sessions placed on it at most; `None`, no bound.
    pub max_agents: Option<usize>,
    pub engine: Arc<Engine>,
    pub stt: Option<Speech>,
    pub tts: Option<Speech>,
    judged: Mutex<Judged>,
    /// Whether its verdict is healthy, and whether it was available at its last check.
    healthy: AtomicBool,
    was_available: AtomicBool,
    /// Its requests in flight, by class.
    in_flight: [AtomicU32; 3],
    /// Its interactive and background requests in flight: what the failover's bound counts.
    agents: AtomicU32,
    series: Series,
}

impl Node {
    /// Whether it takes new sessions and requests: healthy, and its engine ready.
    pub fn available(&self) -> bool {
        self.healthy.load(Ordering::Acquire) && self.engine.upstream.ready()
    }

    /// Whether it has the role that sessions of `class` need.
    pub fn takes(&self, class: SloClass) -> bool {
        match class {
            SloClass::Realtime => self.realtime,
            SloClass::Interactive | SloClass::Background => self.background,
        }
    }

    pub fn verdict(&self) -> Verdict {
        self.lock().verdict
    }

    fn in_flight(&self) -> u32 {
        self.in_flight
            .iter()
            .map(|n| n.load(Ordering::Relaxed))
            .sum()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Judged> {
        self.judged.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The nodes and the sessions placed on them.
#[derive(Debug)]
pub(crate) struct Nodes {
    pub list: Vec<Arc<Node>>,
    table: RwLock<Table>,
    pub config: Placement,
    metrics: metrics::Metrics,
    /// The first node with the realtime role, in the configuration's order: with `calls = "fill"` the one whose own
    /// calls never make it cover.
    first_realtime: Option<usize>,
    /// Woken whenever a node may have stopped covering (the table or a node's availability changed) and when an
    /// agent's request ends: what the requests held back by a failover wait on.
    changed: Notify,
}

/// A request's place on its node, from its placement to its end: dropping it ends the request there.
#[derive(Debug)]
pub(crate) struct Placed {
    nodes: Arc<Nodes>,
    pub node: Arc<Node>,
    class: SloClass,
    /// Its session as the table keys it; `None` for a session the gateway minted, which no later request names.
    key: Option<Key>,
    trusted: bool,
    /// Its session's placement, when the table remembers it.
    session: Option<Arc<Session>>,
    /// Whether it counts among its node's requests in flight: from [`Nodes::admit`] on.
    started: bool,
}

impl Drop for Placed {
    fn drop(&mut self) {
        if let Some(session) = &self.session {
            session.ended(Instant::now());
        }
        if self.started {
            let class = self.class as usize;
            self.node.in_flight[class].fetch_sub(1, Ordering::Relaxed);
            self.node.series.in_flight[class].dec();
            if self.class != SloClass::Realtime {
                self.node.agents.fetch_sub(1, Ordering::AcqRel);
                self.nodes.changed.notify_waiters();
            }
        }
    }
}

/// Why a request held back by a failover went no further.
#[derive(Debug)]
pub(crate) enum Held {
    /// Its wait reached its deadline.
    Late(Timeout),
    /// The drain's cut.
    Cut,
}

impl Nodes {
    /// The nodes of `config`, each with its engine, its canary and its checks, and the task that ends idle
    /// sessions' placements.
    pub fn start(
        config: &[NodeConfig],
        placement: Placement,
        canary: Option<Canary>,
        signals: PrioritySignals,
        idle_timeout: Duration,
        metrics: &metrics::Metrics,
    ) -> Arc<Self> {
        let list: Vec<Arc<Node>> = config
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let name: Arc<str> = node.name.as_str().into();
                let series = metrics.series(&name);
                let changed = Arc::new(Notify::new());
                let health = Health::new(
                    canary,
                    Watched {
                        gauges: series.engine_state.clone(),
                        probes: series.probes.clone(),
                        changed: Some(Arc::clone(&changed)),
                        node: Some(Arc::clone(&name)),
                    },
                );
                let upstream =
                    Upstream::new(node.upstream.clone(), idle_timeout, series.ready.clone())
                        .of_node(Arc::clone(&name));
                let speech = |addr: &Option<UpstreamAddr>| {
                    addr.as_ref().map(|addr| Speech {
                        addr: addr.clone(),
                        client: Upstream::new(addr.clone(), idle_timeout, Gauge::default())
                            .of_node(Arc::clone(&name)),
                    })
                };
                Arc::new(Node {
                    index,
                    realtime: node.realtime,
                    background: node.background,
                    max_calls: node.max_calls,
                    max_agents: node.max_agents,
                    engine: Arc::new(Engine {
                        upstream,
                        health: Arc::new(health),
                        signals,
                        changed,
                    }),
                    stt: speech(&node.stt),
                    tts: speech(&node.tts),
                    judged: Mutex::new(Judged {
                        verdict: Verdict::Unknown,
                        failures: 0,
                        passes: 0,
                        last: [false; 3],
                    }),
                    healthy: AtomicBool::new(false),
                    was_available: AtomicBool::new(false),
                    in_flight: Default::default(),
                    agents: AtomicU32::new(0),
                    name,
                    series,
                })
            })
            .collect();
        let gauges = list
            .iter()
            .map(|node| node.series.sessions.clone())
            .collect();
        let first_realtime = list
            .iter()
            .find(|node| node.realtime)
            .map(|node| node.index);
        let nodes = Arc::new(Self {
            list,
            table: RwLock::new(Table::new(gauges)),
            config: placement,
            metrics: metrics.clone(),
            first_realtime,
            changed: Notify::new(),
        });
        for node in &nodes.list {
            if canary.is_some() {
                tokio::spawn(canary::run(Arc::downgrade(&node.engine)));
            }
            tokio::spawn(watch(Arc::downgrade(&nodes), node.index));
        }
        tokio::spawn(expire(Arc::downgrade(&nodes), placement.idle));
        nodes
    }

    /// The node named `name`.
    pub fn named(&self, name: &str) -> Option<&Arc<Node>> {
        self.list.iter().find(|node| &*node.name == name)
    }

    /// An available node's engine, the first in the configuration's order: for a request without a session, such
    /// as a model list.
    pub fn any(&self) -> Option<Arc<Engine>> {
        self.list
            .iter()
            .find(|node| node.available())
            .map(|node| Arc::clone(&node.engine))
    }

    /// Whether every role is taken by an available node, `GET /ready`'s answer; else the role that is not.
    pub fn ready(&self) -> Result<(), &'static str> {
        for (role, class) in [
            ("realtime", SloClass::Realtime),
            ("background", SloClass::Background),
        ] {
            if !self.list.iter().any(|n| n.takes(class) && n.available()) {
                return Err(role);
            }
        }
        Ok(())
    }

    /// Drains every engine: their states become `drained`.
    pub fn drain(&self) {
        for node in &self.list {
            node.engine.health.drain();
        }
    }

    /// Places a request of `class`, of a `trusted` tenant or not, whose session is `key`: on its session's node
    /// while that node is available, otherwise on the node chosen for it, which a named session keeps. A request
    /// of a session the gateway minted (`key` `None`) is placed on its own.
    pub fn request(
        self: &Arc<Self>,
        key: Option<Key>,
        class: SloClass,
        trusted: bool,
    ) -> Result<Placed, Unplaced> {
        let available = |node: usize| self.list[node].available();
        if let Some(key) = &key {
            let table = self.read();
            if let Found::On(session) = table.find(key, available) {
                // Under the table's lock, so that the expiry, which writes it, cannot end the placement meanwhile.
                session.began();
                drop(table);
                return Ok(self.placed(
                    session.node,
                    class,
                    Some(key.clone()),
                    trusted,
                    Some(session),
                ));
            }
        }
        let mut table = self.write();
        let found = match &key {
            Some(key) => table.find(key, available),
            None => Found::None,
        };
        let (node, session) = match found {
            // Placed by another of its requests meanwhile.
            Found::On(session) => {
                session.began();
                (session.node, Some(session))
            }
            found => {
                let lost = match &found {
                    Found::Lost(old) => Some(old.node),
                    _ => None,
                };
                let node = self
                    .choose(&table, class, None, key.is_some())
                    .inspect_err(|unplaced| {
                        self.metrics.refused(class, *unplaced);
                    })?;
                let session = key
                    .as_ref()
                    .and_then(|key| table.place(key, class, node, trusted, lost, Instant::now()));
                if let Some(session) = &session {
                    session.began();
                }
                drop(table);
                let reason = if lost.is_some() {
                    Reason::Moved
                } else {
                    Reason::New
                };
                self.placed_on(node, class, reason, lost, key.as_ref());
                (node, session)
            }
        };
        Ok(self.placed(node, class, key, trusted, session))
    }

    /// `POST /v1/sessions`: the node of `key`'s session, placed now as a session of `class` if it has none or its
    /// node is not available.
    pub fn session(
        self: &Arc<Self>,
        key: &Key,
        class: SloClass,
        trusted: bool,
    ) -> Result<Arc<Node>, Unplaced> {
        let mut table = self.write();
        let lost = match table.find(key, |node| self.list[node].available()) {
            Found::On(session) => return Ok(Arc::clone(&self.list[session.node])),
            Found::Lost(old) => Some(old.node),
            Found::None => None,
        };
        let refused = |unplaced: Unplaced| {
            self.metrics.refused(class, unplaced);
            unplaced
        };
        let node = self.choose(&table, class, None, true).map_err(refused)?;
        table
            .place(key, class, node, trusted, lost, Instant::now())
            .ok_or(Unplaced::Full)
            .map_err(refused)?;
        drop(table);
        let reason = if lost.is_some() {
            Reason::Moved
        } else {
            Reason::New
        };
        self.placed_on(node, class, reason, lost, Some(key));
        Ok(Arc::clone(&self.list[node]))
    }

    /// Sends a realtime request whose node failed before its first content to another available node: its session
    /// moves there, counted as retried. `Err` when no other node takes it.
    pub fn retry(self: &Arc<Self>, failed: &Placed) -> Result<Placed, Unplaced> {
        let class = failed.class;
        let from = failed.node.index;
        let mut table = self.write();
        let node = self
            .choose(&table, class, Some(from), failed.key.is_some())
            .inspect_err(|unplaced| self.metrics.refused(class, *unplaced))?;
        let session = failed.key.as_ref().and_then(|key| {
            table.place(key, class, node, failed.trusted, Some(from), Instant::now())
        });
        if let Some(session) = &session {
            session.began();
        }
        drop(table);
        failed.node.series.retries.inc();
        self.placed_on(
            node,
            class,
            Reason::Retried,
            Some(from),
            failed.key.as_ref(),
        );
        let mut placed = self.placed(node, class, failed.key.clone(), failed.trusted, session);
        self.count_in(&mut placed);
        Ok(placed)
    }

    /// Counts a request in flight on its node, holding back an interactive or background one while its node covers
    /// ([`Nodes::covering`]): until fewer than `failover_background_max_in_flight` of them are in flight there, the
    /// node no longer covers, its wait reaches `deadline`, or the drain's `cut` ends it. Returns how long it waited.
    pub async fn admit(
        &self,
        placed: &mut Placed,
        (deadline, timeout): (Instant, Timeout),
        mut cut: Option<watch::Receiver<bool>>,
    ) -> Result<Duration, Held> {
        if placed.class == SloClass::Realtime {
            self.count_in(placed);
            return Ok(Duration::ZERO);
        }
        let arrived = Instant::now();
        let mut waiting: Option<GaugeGuard> = None;
        loop {
            // Enabled before the check, so that a change between the two wakes the wait.
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.take_agent_slot(&placed.node) {
                self.count_in(placed);
                return Ok(arrived.elapsed());
            }
            if waiting.is_none() {
                waiting = Some(GaugeGuard::new(placed.node.series.waiting.clone()));
                tracing::debug!(
                    node = &*placed.node.name,
                    "an agent's request waits: the node covers, carrying calls moved to it or a warm spare's"
                );
            }
            tokio::select! {
                biased;
                () = crate::server::cut(&mut cut) => return Err(Held::Cut),
                () = tokio::time::sleep_until(deadline.into()) => return Err(Held::Late(timeout)),
                () = &mut notified => {}
            }
        }
    }

    /// Whether `node` covers, which holds its agents' requests to `failover_background_max_in_flight`: while it
    /// carries realtime sessions moved to it from another node, until they end, whether or not that node is available
    /// again; and, with `calls = "fill"`, while it carries any call and is a realtime node after the first, a warm
    /// spare that took calls only because the nodes before it were full or not available.
    pub fn covering(&self, node: usize) -> bool {
        self.covering_in(&self.read(), node)
    }

    fn covering_in(&self, table: &Table, node: usize) -> bool {
        let moved_in = table.moved_in(node).next().is_some();
        let spare = self.config.calls == CallPlacement::Fill
            && self.list[node].realtime
            && Some(node) != self.first_realtime
            && table.held(node)[0] > 0;
        moved_in || spare
    }

    /// Takes a place among `node`'s agents' requests in flight: always while it does not cover, and only below
    /// `failover_background_max_in_flight` while it does.
    fn take_agent_slot(&self, node: &Node) -> bool {
        if !self.covering(node.index) {
            node.agents.fetch_add(1, Ordering::AcqRel);
            return true;
        }
        let limit =
            u32::try_from(self.config.failover_background_max_in_flight).unwrap_or(u32::MAX);
        node.agents
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < limit).then_some(n + 1)
            })
            .is_ok()
    }

    fn count_in(&self, placed: &mut Placed) {
        let class = placed.class as usize;
        placed.node.in_flight[class].fetch_add(1, Ordering::Relaxed);
        placed.node.series.in_flight[class].inc();
        placed.started = true;
    }

    /// A request's place on `node`, of `session`, whose request the caller counted (under the table's lock).
    fn placed(
        self: &Arc<Self>,
        node: usize,
        class: SloClass,
        key: Option<Key>,
        trusted: bool,
        session: Option<Arc<Session>>,
    ) -> Placed {
        Placed {
            nodes: Arc::clone(self),
            node: Arc::clone(&self.list[node]),
            class,
            key,
            trusted,
            session,
            started: false,
        }
    }

    /// Counts a placement on `node`, logs a move from `from`, and shows each node's covering: a move can make a node
    /// cover or end another's, and so can a new call on a warm spare.
    fn placed_on(
        &self,
        node: usize,
        class: SloClass,
        reason: Reason,
        from: Option<usize>,
        key: Option<&Key>,
    ) {
        let to = &self.list[node];
        to.series.placements[class as usize][reason as usize].inc();
        if let Some(from) = from {
            let session = key.map(|key| key.session.as_str());
            let from = &*self.list[from].name;
            match reason {
                Reason::Retried => tracing::warn!(
                    session,
                    from,
                    to = &*to.name,
                    "a realtime request is sent again on another node: its node failed before its first content"
                ),
                _ => tracing::info!(
                    session,
                    from,
                    to = &*to.name,
                    class = class.as_str(),
                    "a session moves to another node: its node is not available"
                ),
            }
        }
        self.show_covering();
    }

    /// The node a new session of `class` goes to, not `exclude`: of the available nodes with its role, for a
    /// realtime session, among those with a free call slot, the one with the fewest calls, then the fewest requests
    /// in flight (`calls = "spread"`), or the first (`"fill"`); and for an agent's, among those with a free agent slot
    /// (below their `max_agents`, if they have one), one that does not cover, then the one with the fewest agents'
    /// sessions, then the fewest requests in flight. Ties go to the earlier node. A request that names no session
    /// (`named` false) takes no agent slot, so a node with a count of agents takes it only while that node holds no
    /// call.
    fn choose(
        &self,
        table: &Table,
        class: SloClass,
        exclude: Option<usize>,
        named: bool,
    ) -> Result<usize, Unplaced> {
        let mut candidates = self
            .list
            .iter()
            .filter(|node| Some(node.index) != exclude && node.takes(class) && node.available())
            .peekable();
        if candidates.peek().is_none() {
            return Err(Unplaced::NoNode);
        }
        let chosen = if class == SloClass::Realtime {
            let mut free =
                candidates.filter(|node| (table.held(node.index)[0] as usize) < node.max_calls);
            let chosen = match self.config.calls {
                CallPlacement::Spread => free
                    .min_by_key(|node| (table.held(node.index)[0], node.in_flight(), node.index)),
                // The list keeps the configuration's order.
                CallPlacement::Fill => free.next(),
            };
            chosen.ok_or(Unplaced::NoCallSlot)?
        } else {
            candidates
                .filter(|node| {
                    node.max_agents.is_none_or(|max| {
                        (table.agents(node.index) as usize) < max
                            && (named || table.held(node.index)[0] == 0)
                    })
                })
                .min_by_key(|node| {
                    (
                        self.covering_in(table, node.index),
                        table.agents(node.index),
                        node.in_flight(),
                        node.index,
                    )
                })
                .ok_or(Unplaced::NoAgentSlot)?
        };
        Ok(chosen.index)
    }

    /// Ends the placements of the sessions that `tenant`'s `session_closed` events close.
    pub fn closed(&self, tenant: &Tenant, events: &[LifecycleEvent]) {
        let closed: Vec<&SessionId> = events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::SessionClosed))
            .map(|event| &event.session_id)
            .collect();
        if closed.is_empty() {
            return;
        }
        let mut table = self.write();
        for session in closed {
            table.forget(&Key {
                tenant: tenant.index,
                session: session.clone(),
            });
        }
        drop(table);
        self.show_covering();
    }

    /// `DELETE /v1/sessions/{id}`: ends the placement of `key`'s session, as its `session_closed` event does, and
    /// returns the node it was on; `None` for a session without one (never placed, or ended already).
    pub fn end(&self, key: &Key) -> Option<Arc<Node>> {
        let ended = self.write().forget(key)?;
        self.show_covering();
        Some(Arc::clone(&self.list[ended.node]))
    }

    /// Judges node `index` by a check of its components (engine, STT, TTS), logging a change of verdict, and wakes
    /// the requests a failover holds back when its availability changed.
    fn judge(&self, index: usize, checks: [bool; 3]) {
        let node = &self.list[index];
        for (gauge, passed) in node.series.components.iter().zip(checks) {
            gauge.set(i64::from(passed));
        }
        let (before, after) = {
            let mut judged = node.lock();
            let before = judged.verdict;
            judged.last = checks;
            let passed = checks.iter().all(|passed| *passed);
            judged.record(
                passed,
                self.config.unhealthy_after,
                self.config.healthy_after,
            );
            (before, judged.verdict)
        };
        let healthy = after == Verdict::Healthy;
        node.healthy.store(healthy, Ordering::Release);
        node.series.healthy.set(i64::from(healthy));
        if before != after {
            let failed: Vec<&str> = metrics::COMPONENTS
                .iter()
                .zip(checks)
                .filter(|(_, passed)| !passed)
                .map(|(component, _)| *component)
                .collect();
            match after {
                Verdict::Healthy => tracing::info!(node = &*node.name, "the node is healthy"),
                Verdict::Unhealthy => tracing::warn!(
                    node = &*node.name,
                    failed = ?failed,
                    "the node is unhealthy: it takes no new sessions, and its sessions move by their next request"
                ),
                Verdict::Unknown => {}
            }
        }
        let available = node.available();
        if node.was_available.swap(available, Ordering::AcqRel) != available {
            self.show_covering();
        }
    }

    /// Sets each node's covering gauge, and wakes the requests a failover holds back, since a node may have stopped
    /// covering: after each change of the table, and of a node's availability.
    fn show_covering(&self) {
        {
            let table = self.read();
            for node in &self.list {
                node.series
                    .covering
                    .set(i64::from(self.covering_in(&table, node.index)));
            }
        }
        self.changed.notify_waiters();
    }

    /// What a reload's successor takes over: each node's verdict and every placement, by name.
    pub fn handover(&self, tenants: &Tenants) -> Handed {
        let table = self.read();
        let name = |node: usize| self.list[node].name.to_string();
        Handed {
            nodes: self
                .list
                .iter()
                .map(|node| HandedNode {
                    name: node.name.to_string(),
                    verdict: node.verdict(),
                })
                .collect(),
            sessions: table
                .each()
                .filter_map(|(key, session)| {
                    Some(HandedSession {
                        tenant: tenants.all().get(key.tenant)?.name.to_string(),
                        session: key.session.clone(),
                        node: name(session.node),
                        class: session.class,
                        trusted: session.trusted,
                        moved_from: session.moved_from.map(name),
                    })
                })
                .collect(),
        }
    }

    /// Takes over what a reload's predecessor handed: the verdicts of the nodes not judged yet, and the placements
    /// on nodes and of tenants this configuration still has.
    pub fn restore(&self, handed: &Handed, tenants: &Tenants) {
        let index = |name: &str| self.list.iter().position(|node| &*node.name == name);
        for node in &handed.nodes {
            let Some(i) = index(&node.name) else {
                continue;
            };
            let target = &self.list[i];
            let mut judged = target.lock();
            if judged.verdict == Verdict::Unknown && node.verdict != Verdict::Unknown {
                judged.verdict = node.verdict;
                let healthy = node.verdict == Verdict::Healthy;
                target.healthy.store(healthy, Ordering::Release);
                target.series.healthy.set(i64::from(healthy));
            }
        }
        let mut table = self.write();
        let now = Instant::now();
        let mut taken = 0usize;
        for session in &handed.sessions {
            let (Some(node), Some(tenant)) = (index(&session.node), tenants.index(&session.tenant))
            else {
                continue;
            };
            let key = Key {
                tenant,
                session: session.session.clone(),
            };
            if matches!(table.find(&key, |_| true), Found::None) {
                let moved_from = session.moved_from.as_deref().and_then(index);
                let placed =
                    table.place(&key, session.class, node, session.trusted, moved_from, now);
                taken += usize::from(placed.is_some());
            }
        }
        drop(table);
        for node in &self.list {
            node.was_available
                .store(node.available(), Ordering::Release);
        }
        self.show_covering();
        tracing::info!(
            sessions = taken,
            "took over the predecessor's placements and its nodes' verdicts"
        );
    }

    /// `GET /nodes`' body.
    pub fn describe(&self) -> serde_json::Value {
        let table = self.read();
        let by_class = |counts: [u32; 3]| {
            serde_json::json!({
                "realtime": counts[0],
                "interactive": counts[1],
                "background": counts[2],
            })
        };
        let nodes: Vec<serde_json::Value> = self
            .list
            .iter()
            .map(|node| {
                let judged = node.lock();
                let [engine, stt, tts] = judged.last;
                let speech = |speech: &Option<Speech>, check: bool| {
                    speech.as_ref().map(|speech| {
                        serde_json::json!({"address": speech.addr.to_string(), "check": check})
                    })
                };
                let mut roles = Vec::new();
                if node.realtime {
                    roles.push("realtime");
                }
                if node.background {
                    roles.push("background");
                }
                serde_json::json!({
                    "name": &*node.name,
                    "roles": roles,
                    "verdict": judged.verdict.as_str(),
                    "available": node.available(),
                    "engine": {
                        "address": node.engine.upstream.addr().to_string(),
                        "state": node.engine.health.state().as_str(),
                        "ready": node.engine.upstream.ready(),
                        "check": engine,
                    },
                    "stt": speech(&node.stt, stt),
                    "tts": speech(&node.tts, tts),
                    "max_calls": node.max_calls,
                    "max_agents": node.max_agents,
                    "covering": self.covering_in(&table, node.index),
                    "sessions": by_class(table.held(node.index)),
                    "in_flight": by_class(node.in_flight.each_ref().map(|n| n.load(Ordering::Relaxed))),
                })
            })
            .collect();
        serde_json::json!({
            "nodes": nodes,
            "sessions": table.len(),
            "placement": {
                "calls": self.config.calls,
                "failover_background_max_in_flight": self.config.failover_background_max_in_flight,
            },
        })
    }

    fn read(&self) -> RwLockReadGuard<'_, Table> {
        self.table.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Table> {
        self.table.write().unwrap_or_else(PoisonError::into_inner)
    }
}

/// What a reload's predecessor hands its successor, in the ledger's closing line.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Handed {
    pub nodes: Vec<HandedNode>,
    pub sessions: Vec<HandedSession>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct HandedNode {
    pub name: String,
    pub verdict: Verdict,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct HandedSession {
    pub tenant: String,
    pub session: SessionId,
    pub node: String,
    pub class: SloClass,
    pub trusted: bool,
    #[serde(default)]
    pub moved_from: Option<String>,
}

/// Judges node `index` every `health_interval_s`, and at once when its engine's state or readiness changes, while
/// the gateway runs.
async fn watch(nodes: Weak<Nodes>, index: usize) {
    let (node, interval, timeout) = match nodes.upgrade() {
        Some(nodes) => (
            Arc::clone(&nodes.list[index]),
            nodes.config.health_interval,
            nodes.config.health_timeout,
        ),
        None => return,
    };
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        {
            let notified = node.engine.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            tokio::select! {
                _ = ticks.tick() => {}
                () = &mut notified => {}
            }
        }
        let engine = &node.engine;
        let serving = engine.health.state() == EngineState::Serving && engine.upstream.ready();
        let (stt, tts) = tokio::join!(
            answers(node.stt.as_ref(), timeout),
            answers(node.tts.as_ref(), timeout)
        );
        let Some(nodes) = nodes.upgrade() else {
            return;
        };
        nodes.judge(index, [serving, stt, tts]);
    }
}

/// Whether a speech server answers its `/health` within `timeout`; a node without one has nothing to check.
async fn answers(speech: Option<&Speech>, timeout: Duration) -> bool {
    match speech {
        Some(speech) => speech.client.healthy(timeout).await,
        None => true,
    }
}

/// Ends the placements of the sessions idle for `idle`, until the nodes are gone. Every session idles out `idle`
/// after it last went idle, so none that goes idle while this sleeps can idle out before it wakes.
async fn expire(nodes: Weak<Nodes>, idle: Duration) {
    let mut next = Instant::now() + idle;
    loop {
        tokio::time::sleep_until(next.into()).await;
        let Some(nodes) = nodes.upgrade() else {
            return;
        };
        let now = Instant::now();
        let soonest = nodes.write().expire(now, idle);
        // A call that made its node cover may have ended.
        nodes.show_covering();
        next = soonest.unwrap_or(now + idle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CacheSaltSecret, Tenant as TenantConfig};

    fn tenants(names: &[&str]) -> Tenants {
        let config: Vec<TenantConfig> = names
            .iter()
            .map(|name| TenantConfig {
                name: (*name).to_owned(),
                keys_sha256: Vec::new(),
                trusted: true,
                default_class: SloClass::Realtime,
                max_class: SloClass::Realtime,
                max_in_flight: 8,
            })
            .collect();
        Tenants::new(&config, &CacheSaltSecret([0; 32]))
    }

    fn nodes(names: &[&str]) -> Arc<Nodes> {
        let unbounded: Vec<(&str, Option<usize>)> =
            names.iter().map(|name| (*name, None)).collect();
        bounded(&unbounded)
    }

    /// Nodes of `(name, max_agents)`, each with both roles and two call slots.
    fn bounded(nodes: &[(&str, Option<usize>)]) -> Arc<Nodes> {
        started(nodes, Placement::default())
    }

    /// Nodes a and b, their agents unbounded, that place calls by `calls`.
    fn placing(calls: CallPlacement) -> Arc<Nodes> {
        let placement = Placement {
            calls,
            ..Placement::default()
        };
        started(&[("a", None), ("b", None)], placement)
    }

    /// Nodes of `(name, max_agents)`, each with both roles and two call slots, placed by `placement`.
    fn started(nodes: &[(&str, Option<usize>)], placement: Placement) -> Arc<Nodes> {
        let both: Vec<(&str, bool, Option<usize>)> = nodes
            .iter()
            .map(|(name, max_agents)| (*name, true, *max_agents))
            .collect();
        started_with(&both, placement)
    }

    /// Nodes of `(name, realtime, max_agents)`, each with the background role, the realtime role where `realtime`
    /// says, and two call slots, placed by `placement`.
    fn started_with(nodes: &[(&str, bool, Option<usize>)], placement: Placement) -> Arc<Nodes> {
        let config: Vec<NodeConfig> = nodes
            .iter()
            .map(|(name, realtime, max_agents)| NodeConfig {
                name: (*name).to_owned(),
                upstream: UpstreamAddr::Unix(format!("/nonexistent/{name}.sock").into()),
                realtime: *realtime,
                background: true,
                max_calls: 2,
                max_agents: *max_agents,
                stt: None,
                tts: None,
            })
            .collect();
        let names: Vec<Arc<str>> = nodes.iter().map(|(name, ..)| Arc::from(*name)).collect();
        // Checked rarely, so that the engines nobody serves are not judged during the test.
        let placement = Placement {
            health_interval: Duration::from_secs(3600),
            ..placement
        };
        Nodes::start(
            &config,
            placement,
            None,
            PrioritySignals::default(),
            Duration::from_secs(4),
            &metrics::Metrics::new(&names),
        )
    }

    /// Judges every node by a passed check, as its first check would: each is healthy, and available.
    fn all_healthy(nodes: &Nodes) {
        for node in &nodes.list {
            nodes.judge(node.index, [true; 3]);
        }
    }

    /// Judges node `index` by failed checks of its engine until it is unhealthy, and not available.
    fn fail(nodes: &Nodes, index: usize) {
        for _ in 0..nodes.config.unhealthy_after {
            nodes.judge(index, [false, true, true]);
        }
        assert!(!nodes.list[index].available());
    }

    /// Judges node `index` by passed checks until it is healthy again, and available.
    fn recover(nodes: &Nodes, index: usize) {
        for _ in 0..nodes.config.healthy_after {
            nodes.judge(index, [true; 3]);
        }
        assert!(nodes.list[index].available());
    }

    /// The `session_closed` event of the session `id`, posted by its tenant, the first.
    fn close(nodes: &Nodes, id: &str) {
        let event: LifecycleEvent = serde_json::from_value(serde_json::json!({
            "event_id": format!("close-{id}"),
            "session_id": id,
            "at_ms": 1,
            "source": "voice-worker",
            "type": "session_closed",
        }))
        .unwrap();
        nodes.closed(&tenants(&["voice"]).all()[0], &[event]);
    }

    fn key(id: &str) -> Key {
        Key {
            tenant: 0,
            session: SessionId::new(id).unwrap(),
        }
    }

    /// The node that a request of the session `id`, of `class`, is placed on; the request ends at once.
    fn place(nodes: &Arc<Nodes>, id: &str, class: SloClass) -> Result<String, Unplaced> {
        let placed = nodes.request(Some(key(id)), class, true)?;
        Ok(placed.node.name.to_string())
    }

    /// Where `count` new sessions of `class`, `{prefix}-1` on, are placed, one after another.
    fn place_new(nodes: &Arc<Nodes>, prefix: &str, count: usize, class: SloClass) -> Vec<String> {
        (1..=count)
            .map(|i| place(nodes, &format!("{prefix}-{i}"), class).unwrap())
            .collect()
    }

    fn session(tenant: &str, id: &str, node: &str, moved_from: Option<&str>) -> HandedSession {
        HandedSession {
            tenant: tenant.into(),
            session: SessionId::new(id).unwrap(),
            node: node.into(),
            class: SloClass::Realtime,
            trusted: true,
            moved_from: moved_from.map(Into::into),
        }
    }

    #[tokio::test]
    async fn a_successor_takes_the_placements_it_can_and_the_verdicts_of_nodes_it_has_not_judged() {
        let successor = nodes(&["a", "b"]);
        let tenants = tenants(&["voice"]);
        // The test's runtime runs no other task until it waits, so no node has been checked yet; b is judged here,
        // as if by the successor's own first check.
        successor.list[1].lock().verdict = Verdict::Unhealthy;
        let handed = Handed {
            nodes: vec![
                HandedNode {
                    name: "a".into(),
                    verdict: Verdict::Healthy,
                },
                HandedNode {
                    name: "b".into(),
                    verdict: Verdict::Healthy,
                },
                HandedNode {
                    name: "c".into(),
                    verdict: Verdict::Healthy,
                },
            ],
            sessions: vec![
                session("voice", "call-1", "b", Some("a")),
                // A node or a tenant the successor's configuration no longer has.
                session("voice", "call-2", "c", None),
                session("bench", "call-3", "a", None),
            ],
        };
        successor.restore(&handed, &tenants);
        assert_eq!(
            successor
                .list
                .iter()
                .map(|n| n.verdict())
                .collect::<Vec<_>>(),
            [Verdict::Healthy, Verdict::Unhealthy]
        );
        let back = successor.handover(&tenants);
        assert_eq!(back.sessions, [session("voice", "call-1", "b", Some("a"))]);
        // The call holds one of b's call slots, and b covers while it carries the call moved from a, healthy as a is.
        assert_eq!(successor.read().held(1), [1, 0, 0]);
        assert!(successor.covering(1));
    }

    #[test]
    fn a_node_is_unhealthy_after_n_failures_and_healthy_again_after_m_passes() {
        let mut judged = Judged {
            verdict: Verdict::Unknown,
            failures: 0,
            passes: 0,
            last: [false; 3],
        };
        // Unknown: failures short of unhealthy_after leave it unknown, and its first pass makes it healthy.
        judged.record(false, 3, 2);
        judged.record(false, 3, 2);
        assert_eq!(judged.verdict, Verdict::Unknown);
        judged.record(true, 3, 2);
        assert_eq!(judged.verdict, Verdict::Healthy);
        // Two failures and a pass: still healthy; three in a row: unhealthy.
        for passed in [false, false, true, false, false] {
            judged.record(passed, 3, 2);
        }
        assert_eq!(judged.verdict, Verdict::Healthy);
        judged.record(false, 3, 2);
        assert_eq!(judged.verdict, Verdict::Unhealthy);
        // One pass is not enough, and a failure starts the count again.
        for passed in [true, false, true] {
            judged.record(passed, 3, 2);
            assert_eq!(judged.verdict, Verdict::Unhealthy);
        }
        judged.record(true, 3, 2);
        assert_eq!(judged.verdict, Verdict::Healthy);
    }

    #[tokio::test]
    async fn an_agents_session_goes_past_a_node_at_its_max_agents_to_one_with_a_free_slot() {
        let nodes = bounded(&[("a", Some(2)), ("b", None)]);
        all_healthy(&nodes);
        // The fewest agents' sessions first: a, b, a, b. a then holds its two, so the fifth session, which that order
        // would give a (two each, a the earlier), goes to b, as does every one after it: b has no bound.
        assert_eq!(
            place_new(&nodes, "agent", 6, SloClass::Background),
            ["a", "b", "a", "b", "b", "b"]
        );
        assert_eq!((nodes.read().agents(0), nodes.read().agents(1)), (2, 4));
        // A placed session keeps its node, and the bound is the agents' alone: a call still takes a's call slots.
        assert_eq!(place(&nodes, "agent-1", SloClass::Background).unwrap(), "a");
        assert_eq!(place(&nodes, "call-1", SloClass::Realtime).unwrap(), "a");
        // /nodes shows each node's bound beside its call slots, none as null.
        let view = nodes.describe();
        let bound = |node: usize| view["nodes"][node].get("max_agents").cloned();
        assert_eq!(
            (bound(0), bound(1)),
            (Some(serde_json::json!(2)), Some(serde_json::Value::Null))
        );
    }

    #[tokio::test]
    async fn an_agents_session_that_finds_every_node_at_its_max_agents_is_not_placed() {
        let nodes = bounded(&[("a", Some(1)), ("b", Some(2))]);
        all_healthy(&nodes);
        // Interactive and background sessions count together.
        assert_eq!(
            place(&nodes, "agent-1", SloClass::Interactive).unwrap(),
            "a"
        );
        assert_eq!(place(&nodes, "agent-2", SloClass::Background).unwrap(), "b");
        assert_eq!(
            place(&nodes, "agent-3", SloClass::Interactive).unwrap(),
            "b"
        );
        for class in [SloClass::Interactive, SloClass::Background] {
            assert_eq!(place(&nodes, "agent-4", class), Err(Unplaced::NoAgentSlot));
        }
        // As a request without a session takes no call slot but finds none when every node is full, so for agents.
        assert_eq!(
            nodes.request(None, SloClass::Background, true).err(),
            Some(Unplaced::NoAgentSlot)
        );
        // The placed sessions keep their nodes, and a call still finds a's call slots.
        assert_eq!(
            place(&nodes, "agent-1", SloClass::Interactive).unwrap(),
            "a"
        );
        assert_eq!(place(&nodes, "call-1", SloClass::Realtime).unwrap(), "a");
        // With no available node of the role at all, it is no_node still.
        fail(&nodes, 0);
        fail(&nodes, 1);
        assert_eq!(
            place(&nodes, "agent-4", SloClass::Background),
            Err(Unplaced::NoNode)
        );
    }

    #[tokio::test]
    async fn an_agents_request_that_names_no_session_keeps_off_a_bounded_node_while_it_carries_calls()
     {
        let placement = Placement {
            calls: CallPlacement::Fill,
            ..Placement::default()
        };
        let nodes = started(&[("phastform", Some(4)), ("gray", Some(12))], placement);
        all_healthy(&nodes);
        // Both calls on phastform, as the serving arrangement places them, three agents' sessions beside them and
        // eight on gray.
        assert_eq!(
            place_new(&nodes, "call", 2, SloClass::Realtime),
            ["phastform", "phastform"]
        );
        let placed = place_new(&nodes, "agent", 12, SloClass::Background);
        let first = (1..=12).find(|i| placed[i - 1] == "phastform").unwrap();
        close(&nodes, &format!("agent-{first}"));
        assert_eq!((nodes.read().agents(0), nodes.read().agents(1)), (3, 8));
        // Ten background requests that name no session, all in flight at once: phastform's bound cannot count them,
        // so none runs beside its calls.
        let mut held = Vec::new();
        for _ in 0..10 {
            let mut request = nodes.request(None, SloClass::Background, true).unwrap();
            let deadline = (Instant::now() + Duration::from_secs(5), Timeout::Queue);
            nodes.admit(&mut request, deadline, None).await.unwrap();
            held.push(request);
        }
        assert!(held.iter().all(|placed| &*placed.node.name == "gray"));
        let background = |node: usize| {
            nodes.list[node].in_flight[SloClass::Background as usize].load(Ordering::Relaxed)
        };
        assert_eq!((background(0), background(1)), (0, 10));
        // With gray not available, such a request finds no agent slot, as a session would.
        fail(&nodes, 1);
        assert_eq!(
            nodes.request(None, SloClass::Interactive, true).err(),
            Some(Unplaced::NoAgentSlot)
        );
        // Once phastform carries no call it takes them again, below its bound.
        close(&nodes, "call-1");
        close(&nodes, "call-2");
        let request = nodes.request(None, SloClass::Background, true).unwrap();
        assert_eq!(&*request.node.name, "phastform");
    }

    #[tokio::test]
    async fn an_agents_session_moved_from_a_failed_node_needs_a_free_agent_slot_on_its_new_node() {
        let nodes = bounded(&[("a", Some(1)), ("b", Some(1))]);
        all_healthy(&nodes);
        assert_eq!(
            place_new(&nodes, "agent", 2, SloClass::Background),
            ["a", "b"]
        );
        fail(&nodes, 0);
        // agent-1's node is not available, and b holds its one agent's session: the request is not placed, and the
        // session keeps its old placement until a request of it is.
        assert_eq!(
            place(&nodes, "agent-1", SloClass::Background),
            Err(Unplaced::NoAgentSlot)
        );
        assert_eq!(nodes.read().agents(0), 1);
        // b's session ends: agent-1's next request moves it there, counted.
        nodes.write().forget(&key("agent-2"));
        assert_eq!(place(&nodes, "agent-1", SloClass::Background).unwrap(), "b");
        assert_eq!((nodes.read().agents(0), nodes.read().agents(1)), (0, 1));
        let moved = &nodes.list[1].series.placements[SloClass::Background as usize];
        assert_eq!(moved[Reason::Moved as usize].get(), 1);
    }

    #[tokio::test]
    async fn an_agents_session_that_idles_out_frees_its_agent_slot() {
        let nodes = bounded(&[("a", Some(1))]);
        all_healthy(&nodes);
        assert_eq!(place(&nodes, "agent-1", SloClass::Background).unwrap(), "a");
        assert_eq!(
            place(&nodes, "agent-2", SloClass::Background),
            Err(Unplaced::NoAgentSlot)
        );
        // idle_s after its last request ended, the expiry forgets agent-1, and its slot is free.
        let idle = nodes.config.idle;
        nodes.write().expire(Instant::now() + idle, idle);
        assert_eq!(place(&nodes, "agent-2", SloClass::Background).unwrap(), "a");
    }

    #[tokio::test]
    async fn calls_fill_the_nodes_in_order_with_fill_and_spread_over_them_by_default() {
        // fill: the first node until its two call slots are taken, and only then the next.
        let fill = placing(CallPlacement::Fill);
        all_healthy(&fill);
        assert_eq!(
            place_new(&fill, "call", 4, SloClass::Realtime),
            ["a", "a", "b", "b"]
        );
        assert_eq!(
            place(&fill, "call-5", SloClass::Realtime),
            Err(Unplaced::NoCallSlot)
        );
        // spread, the default: the fewest calls first, the earlier node on a tie.
        let spread = nodes(&["a", "b"]);
        assert_eq!(spread.config.calls, CallPlacement::Spread);
        all_healthy(&spread);
        assert_eq!(
            place_new(&spread, "call", 4, SloClass::Realtime),
            ["a", "b", "a", "b"]
        );
        assert_eq!(
            place(&spread, "call-5", SloClass::Realtime),
            Err(Unplaced::NoCallSlot)
        );
        // Agents' sessions spread where no node covers; with fill b, a spare that carries calls, covers, and they go
        // to a.
        assert_eq!(
            place_new(&spread, "agent", 3, SloClass::Background),
            ["a", "b", "a"]
        );
        assert_eq!(
            place_new(&fill, "agent", 3, SloClass::Background),
            ["a", "a", "a"]
        );
    }

    #[tokio::test]
    async fn with_fill_a_call_goes_to_the_next_node_while_the_first_is_not_available() {
        let nodes = placing(CallPlacement::Fill);
        all_healthy(&nodes);
        assert_eq!(place(&nodes, "call-1", SloClass::Realtime).unwrap(), "a");
        fail(&nodes, 0);
        // A new call goes to b, and so does the call whose node failed, which b then covers for.
        assert_eq!(place(&nodes, "call-2", SloClass::Realtime).unwrap(), "b");
        assert_eq!(place(&nodes, "call-1", SloClass::Realtime).unwrap(), "b");
        assert!(nodes.covering(1));
        // a is available again: new calls go to it first, and the moved call stays where it is.
        for _ in 0..nodes.config.healthy_after {
            nodes.judge(0, [true; 3]);
        }
        assert_eq!(place(&nodes, "call-3", SloClass::Realtime).unwrap(), "a");
        assert_eq!(place(&nodes, "call-1", SloClass::Realtime).unwrap(), "b");
    }

    #[tokio::test]
    async fn a_node_covers_while_it_carries_calls_moved_to_it_after_their_node_recovers() {
        let nodes = nodes(&["a", "b"]);
        all_healthy(&nodes);
        assert_eq!(place_new(&nodes, "call", 2, SloClass::Realtime), ["a", "b"]);
        fail(&nodes, 0);
        assert_eq!(place(&nodes, "call-1", SloClass::Realtime).unwrap(), "b");
        assert!(nodes.covering(1));
        // a is available again, and b carries the moved call still: it covers until that call ends.
        recover(&nodes, 0);
        assert!(nodes.covering(1));
        // b's own call ending changes nothing; the moved call's session_closed ends the covering.
        close(&nodes, "call-2");
        assert!(nodes.covering(1));
        close(&nodes, "call-1");
        assert!(!nodes.covering(1));
        // The same for a moved call that idles out.
        assert_eq!(place(&nodes, "call-3", SloClass::Realtime).unwrap(), "a");
        fail(&nodes, 0);
        assert_eq!(place(&nodes, "call-3", SloClass::Realtime).unwrap(), "b");
        recover(&nodes, 0);
        assert!(nodes.covering(1));
        let idle = nodes.config.idle;
        nodes.write().expire(Instant::now() + idle, idle);
        assert!(!nodes.covering(1));
        assert!(!nodes.covering(0));
    }

    #[tokio::test]
    async fn with_fill_a_spare_covers_while_it_carries_a_new_call_until_the_call_ends() {
        let nodes = placing(CallPlacement::Fill);
        all_healthy(&nodes);
        fail(&nodes, 0);
        // A new call while a is down goes to b, which covers though no call moved there.
        assert_eq!(place(&nodes, "call-1", SloClass::Realtime).unwrap(), "b");
        assert_eq!(nodes.read().moved_in(1).count(), 0);
        assert!(nodes.covering(1));
        // a recovers and takes the next call, and b covers until its own call ends.
        recover(&nodes, 0);
        assert_eq!(place(&nodes, "call-2", SloClass::Realtime).unwrap(), "a");
        assert!(nodes.covering(1));
        close(&nodes, "call-1");
        assert!(!nodes.covering(1));
    }

    #[tokio::test]
    async fn with_fill_the_first_realtime_node_never_covers_for_its_own_calls() {
        // g takes no calls, so a is the first realtime node.
        let placement = Placement {
            calls: CallPlacement::Fill,
            ..Placement::default()
        };
        let nodes = started_with(
            &[("g", false, None), ("a", true, None), ("b", true, None)],
            placement,
        );
        all_healthy(&nodes);
        assert_eq!(
            place_new(&nodes, "call", 3, SloClass::Realtime),
            ["a", "a", "b"]
        );
        // a carries two calls and does not cover; b, which took the third because a was full, does.
        let covering: Vec<bool> = (0..3).map(|node| nodes.covering(node)).collect();
        assert_eq!(covering, [false, false, true]);
        // a's agents are bounded by max_agents alone: new agents go to the nodes that do not cover.
        assert_eq!(
            place_new(&nodes, "agent", 4, SloClass::Background),
            ["g", "a", "g", "a"]
        );
    }

    #[tokio::test]
    async fn with_spread_a_nodes_own_calls_never_make_it_cover() {
        let nodes = nodes(&["a", "b"]);
        all_healthy(&nodes);
        assert_eq!(
            place_new(&nodes, "call", 4, SloClass::Realtime),
            ["a", "b", "a", "b"]
        );
        assert!(!nodes.covering(0) && !nodes.covering(1));
        // Nor through a failure and a recovery of the other node, so long as no call moved.
        fail(&nodes, 0);
        recover(&nodes, 0);
        assert!(!nodes.covering(0) && !nodes.covering(1));
    }
}
