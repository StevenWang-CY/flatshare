//! The series of a gateway with `[[nodes]]`, registered in place of the one engine's: its engine's readiness, state
//! and canaries by node, under the same names with a `node` label, and the nodes' health, sessions, requests and
//! placements. Every series exists from the start, at zero.

use std::sync::Arc;

use flatshare_proto::SloClass;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::{Registry, Unit};

use crate::canary::Probes;
use crate::metrics::LATENCY_BUCKETS;
use crate::types::EngineState;

/// What a node's health is made of, as the `component` label: its engine (serving and ready), its STT and its TTS
/// server (answering `/health`).
pub(crate) const COMPONENTS: [&str; 3] = ["engine", "stt", "tts"];

/// Why a session was placed, as the `reason` label: a session without a node (new, or idled out), one moved because
/// its node was not available, and one whose request was sent again on another node after its node failed before the
/// request's first content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reason {
    New,
    Moved,
    Retried,
}

impl Reason {
    pub const ALL: [Self; 3] = [Self::New, Self::Moved, Self::Retried];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Moved => "moved",
            Self::Retried => "retried",
        }
    }
}

/// Why a session found no node, as the `reason` label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unplaced {
    /// No available node takes the session's class.
    NoNode,
    /// Every available node that takes realtime sessions has `max_calls` of them.
    NoCallSlot,
    /// Every available node that takes agents' sessions has a `max_agents` and that many of them.
    NoAgentSlot,
    /// `POST /v1/sessions` found the table full, every session in it with a request in flight.
    Full,
}

impl Unplaced {
    pub const ALL: [Self; 4] = [
        Self::NoNode,
        Self::NoCallSlot,
        Self::NoAgentSlot,
        Self::Full,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoNode => "no_node",
            Self::NoCallSlot => "no_call_slot",
            Self::NoAgentSlot => "no_agent_slot",
            Self::Full => "table_full",
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct NodeLabels {
    node: Arc<str>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct NodeStateLabels {
    node: Arc<str>,
    state: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct NodeResultLabels {
    node: Arc<str>,
    result: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct NodeComponentLabels {
    node: Arc<str>,
    component: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct NodeClassLabels {
    node: Arc<str>,
    class: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct PlacementLabels {
    node: Arc<str>,
    class: &'static str,
    reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RefusedLabels {
    class: &'static str,
    reason: &'static str,
}

/// The node families.
#[derive(Clone, Debug)]
pub(crate) struct Metrics {
    ready: Family<NodeLabels, Gauge>,
    engine_state: Family<NodeStateLabels, Gauge>,
    canary: Family<NodeResultLabels, Counter>,
    canary_seconds: Family<NodeLabels, Histogram, fn() -> Histogram>,
    healthy: Family<NodeLabels, Gauge>,
    components: Family<NodeComponentLabels, Gauge>,
    sessions: Family<NodeClassLabels, Gauge>,
    in_flight: Family<NodeClassLabels, Gauge>,
    covering: Family<NodeLabels, Gauge>,
    draining: Family<NodeLabels, Gauge>,
    waiting: Family<NodeLabels, Gauge>,
    placements: Family<PlacementLabels, Counter>,
    refused: Family<RefusedLabels, Counter>,
    retries: Family<NodeLabels, Counter>,
}

/// One node's series, resolved at start.
#[derive(Clone, Debug)]
pub(crate) struct Series {
    /// Its engine's readiness: 0 while a request found it broken and its `/health` has not answered since.
    pub ready: Gauge,
    pub engine_state: [Gauge; EngineState::ALL.len()],
    pub probes: Probes,
    /// 1 while the node is healthy.
    pub healthy: Gauge,
    /// Each component's last check, in [`COMPONENTS`] order.
    pub components: [Gauge; COMPONENTS.len()],
    /// Its sessions and its requests in flight, by class.
    pub sessions: [Gauge; 3],
    pub in_flight: [Gauge; 3],
    /// 1 while it covers: it carries realtime sessions moved to it from another node, or, as a warm spare, any call.
    pub covering: Gauge,
    /// 1 while it drains, as the configuration sets it.
    pub draining: Gauge,
    /// Its interactive and background requests waiting for the failover's bound.
    pub waiting: Gauge,
    /// Its placements by class and reason, in [`Reason::ALL`] order.
    pub placements: [[Counter; Reason::ALL.len()]; 3],
    /// The realtime requests sent again after a failure of this node before their first content.
    pub retries: Counter,
}

impl Metrics {
    /// The families of the nodes named `names`, every series at zero.
    pub fn new(names: &[Arc<str>]) -> Self {
        let metrics = Self {
            ready: Family::default(),
            engine_state: Family::default(),
            canary: Family::default(),
            canary_seconds: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            healthy: Family::default(),
            components: Family::default(),
            sessions: Family::default(),
            in_flight: Family::default(),
            covering: Family::default(),
            draining: Family::default(),
            waiting: Family::default(),
            placements: Family::default(),
            refused: Family::default(),
            retries: Family::default(),
        };
        for name in names {
            drop(metrics.series(name));
        }
        for class in SloClass::ALL.map(SloClass::as_str) {
            for reason in Unplaced::ALL.map(Unplaced::as_str) {
                drop(
                    metrics
                        .refused
                        .get_or_create(&RefusedLabels { class, reason }),
                );
            }
        }
        metrics
    }

    /// The series of the node `name`.
    pub fn series(&self, name: &Arc<str>) -> Series {
        let node = || Arc::clone(name);
        let by_class = |family: &Family<NodeClassLabels, Gauge>| {
            SloClass::ALL.map(|class| {
                family.get_or_create_owned(&NodeClassLabels {
                    node: node(),
                    class: class.as_str(),
                })
            })
        };
        let result = |result| {
            self.canary.get_or_create_owned(&NodeResultLabels {
                node: node(),
                result,
            })
        };
        Series {
            ready: self.ready.get_or_create_owned(&NodeLabels { node: node() }),
            engine_state: EngineState::ALL.map(|state| {
                self.engine_state.get_or_create_owned(&NodeStateLabels {
                    node: node(),
                    state: state.as_str(),
                })
            }),
            probes: Probes {
                success: result("success"),
                failure: result("failure"),
                seconds: self
                    .canary_seconds
                    .get_or_create_owned(&NodeLabels { node: node() }),
            },
            healthy: self
                .healthy
                .get_or_create_owned(&NodeLabels { node: node() }),
            components: COMPONENTS.map(|component| {
                self.components.get_or_create_owned(&NodeComponentLabels {
                    node: node(),
                    component,
                })
            }),
            sessions: by_class(&self.sessions),
            in_flight: by_class(&self.in_flight),
            covering: self
                .covering
                .get_or_create_owned(&NodeLabels { node: node() }),
            draining: self
                .draining
                .get_or_create_owned(&NodeLabels { node: node() }),
            waiting: self
                .waiting
                .get_or_create_owned(&NodeLabels { node: node() }),
            placements: SloClass::ALL.map(|class| {
                Reason::ALL.map(|reason| {
                    self.placements.get_or_create_owned(&PlacementLabels {
                        node: node(),
                        class: class.as_str(),
                        reason: reason.as_str(),
                    })
                })
            }),
            retries: self
                .retries
                .get_or_create_owned(&NodeLabels { node: node() }),
        }
    }

    /// Counts a session of `class` that found no node.
    pub fn refused(&self, class: SloClass, unplaced: Unplaced) {
        self.refused
            .get_or_create(&RefusedLabels {
                class: class.as_str(),
                reason: unplaced.as_str(),
            })
            .inc();
    }

    /// Registers the families where the one engine's go.
    pub fn register(&self, r: &mut Registry) {
        r.register(
            "node_ready",
            "Whether each node's engine is ready (1), or a request found it broken and its /health has not \
             answered since (0), while the node takes no requests",
            self.ready.clone(),
        );
        r.register(
            "engine_state",
            "Each node's engine's state as inference finds it, 1 for the current one: warming until an inference \
             succeeds, serving, degraded after the canary failed degraded_after times in a row, drained once the \
             gateway drains",
            self.engine_state.clone(),
        );
        r.register(
            "canary_probes",
            "Inference canaries by node and result: a one-token generation the gateway sends a node's engine when \
             nothing succeeded on it within its interval",
            self.canary.clone(),
        );
        r.register_with_unit(
            "canary",
            "Time an inference canary took, to its answer's end or its failure, by node",
            Unit::Seconds,
            self.canary_seconds.clone(),
        );
        r.register(
            "node_healthy",
            "Whether each node is healthy (1): its engine serving and ready and its speech servers answering, \
             unhealthy_after failed checks in a row making it unhealthy and healthy_after passed ones healthy again",
            self.healthy.clone(),
        );
        r.register(
            "node_component_healthy",
            "Each node component's last check: the engine (serving and ready), the STT and the TTS server \
             (answering /health)",
            self.components.clone(),
        );
        r.register(
            "node_sessions",
            "The sessions placed on each node, by the class they were placed with",
            self.sessions.clone(),
        );
        r.register(
            "node_in_flight",
            "Inference requests in flight on each node, by class",
            self.in_flight.clone(),
        );
        r.register(
            "node_covering",
            "Whether each node covers (1), which holds its interactive and background requests to \
             failover_background_max_in_flight: it carries realtime sessions moved to it from another node, until they \
             end, or, with calls set to fill, any call on a realtime node after the first",
            self.covering.clone(),
        );
        r.register(
            "node_draining",
            "Whether each node drains (1), as its draining key sets it: it takes no new session, keeps its calls \
             until they end, and its agents' sessions move by their next request",
            self.draining.clone(),
        );
        r.register(
            "node_failover_waiting",
            "Interactive and background requests waiting on each node for the failover's bound",
            self.waiting.clone(),
        );
        r.register(
            "placements",
            "Sessions placed, by node, class and reason: new (no node yet, or idled out), moved (its node was not \
             available) or retried (its request sent again here after its node failed before the first content)",
            self.placements.clone(),
        );
        r.register(
            "placements_refused",
            "Sessions no node took, by class and reason: no_node (no available node of its role), no_call_slot \
             (every available realtime node at max_calls), no_agent_slot (every available background node at \
             max_agents) or table_full",
            self.refused.clone(),
        );
        r.register(
            "node_retries",
            "Realtime requests sent again on another node after this node failed before their first content",
            self.retries.clone(),
        );
    }
}
