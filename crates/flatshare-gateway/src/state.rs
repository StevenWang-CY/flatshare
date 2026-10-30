//! The state every listener's routes share ([`Shared`]), and each listener's own ([`App`]).

use std::sync::Arc;
use std::time::Duration;

use flatshare_proto::{LifecycleEvent, SloClass};
use tokio::sync::Notify;

use crate::auth::{Tenant, Tenants};
use crate::config::{ClassConfig, MetricsKey, PrioritySignals};
use crate::drain::Drain;
use crate::engine::Engine;
use crate::events::Events;
use crate::harness::Admission;
use crate::intake::Intake;
use crate::ledger::Load;
use crate::metrics::Metrics;
use crate::nodes::Nodes;

/// What the gateway sends its requests to.
#[derive(Debug)]
pub(crate) enum Engines {
    /// One `upstream`, served as it always was.
    One(Arc<Engine>),
    /// `[[nodes]]`, each with its engine, and the sessions placed on them.
    Nodes(Arc<Nodes>),
}

/// State shared by every listener.
#[derive(Debug)]
pub(crate) struct Shared {
    pub tenants: Tenants,
    /// The engine, or the nodes.
    pub engines: Engines,
    /// The bounds on the requests in flight and their bodies, whatever the admission policy.
    pub intake: Arc<Intake>,
    /// The admission point; `None` under policy "none", where requests go upstream at once.
    pub admission: Option<Arc<Admission>>,
    /// Told of every change to what the intake and the admission point hold, for a successor's ledger.
    pub changed: Arc<Notify>,
    pub upstream_priority: PrioritySignals,
    pub metrics: Arc<Metrics>,
    pub events: Events,
    pub max_body_bytes: usize,
    /// What each class's requests may ask for, by [`SloClass`].
    pub classes: [ClassConfig; 3],
    /// The choices a request may ask for at most.
    pub max_choices: u32,
    pub stream_buffer_frames: usize,
    pub stall_timeout: Duration,
    pub drain: Drain,
    /// What `GET /metrics` asks of a request on a TCP listener.
    pub metrics_key: Option<MetricsKey>,
}

impl Shared {
    /// What the gateway holds on its engine, with what it counts of a predecessor's: a successor's ledger line.
    pub fn load(&self) -> Load {
        let closed = self.intake.is_closed();
        Load {
            intake: self.intake.held(),
            admission: self
                .admission
                .as_ref()
                .map(|admission| admission.held().into()),
            closed,
            // The placements and the nodes' verdicts, once: they change no more once the gateway admits nothing.
            nodes: self
                .nodes()
                .filter(|_| closed)
                .map(|nodes| nodes.handover(&self.tenants)),
        }
    }

    /// What a reload's predecessor holds now, from its ledger, counted with the gateway's own until it exits.
    pub fn set_predecessor(&self, load: &Load) {
        self.intake.set_predecessor(load.intake.clone());
        if let Some(admission) = &self.admission {
            admission.set_predecessor(load.elsewhere());
        }
        if let (Some(nodes), Some(handed)) = (self.nodes(), &load.nodes) {
            nodes.restore(handed, &self.tenants);
        }
    }

    /// The nodes, with `[[nodes]]`.
    pub fn nodes(&self) -> Option<&Arc<Nodes>> {
        match &self.engines {
            Engines::Nodes(nodes) => Some(nodes),
            Engines::One(_) => None,
        }
    }

    /// Admits nothing more: the gateway handed its sockets to a successor. Its waiting requests get 503, and the
    /// requests already admitted run on.
    pub fn close(&self) {
        self.intake.close();
        if let Some(admission) = &self.admission {
            admission.close();
        }
    }

    /// The configuration of `class`'s requests.
    pub fn class(&self, class: SloClass) -> &ClassConfig {
        &self.classes[class as usize]
    }

    /// Accepted lifecycle events of `tenant`'s sessions, in the order they were received, to the admission point.
    pub fn lifecycle(&self, tenant: &Tenant, events: &[LifecycleEvent]) {
        if let Some(admission) = &self.admission {
            admission.events(tenant, events);
        }
        // A session_closed event ends its session's placement, and frees a call's slot, as DELETE /v1/sessions/{id}
        // does.
        if let Some(nodes) = self.nodes() {
            nodes.closed(tenant, events);
        }
    }
}

/// The state of one listener's routes.
#[derive(Clone, Debug)]
pub(crate) struct App {
    pub shared: Arc<Shared>,
    /// The tenant of requests on this listener that carry no known API key.
    pub listener_tenant: Option<usize>,
    /// Whether the listener is a Unix socket, which its file mode keeps from other users of the host.
    pub private: bool,
}
