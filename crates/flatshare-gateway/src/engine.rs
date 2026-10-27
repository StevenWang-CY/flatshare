//! An engine as the gateway reaches it: its connection pool and whether its requests go to it
//! ([`crate::upstream`]), and its state as inference finds it ([`crate::canary`]). A gateway with one `upstream` has
//! one engine; with `[[nodes]]`, each node has its own ([`crate::nodes`]).

use std::sync::Arc;

use tokio::sync::Notify;

use crate::canary::Health;
use crate::config::PrioritySignals;
use crate::upstream::{Upstream, recover};

/// One engine.
#[derive(Debug)]
pub(crate) struct Engine {
    pub upstream: Upstream,
    /// Its state as inference finds it, which its canary drives.
    pub health: Arc<Health>,
    /// Where its canary's priority goes, as every request's does.
    pub signals: PrioritySignals,
    /// Woken whenever its readiness or its state changes, so that its node is judged again at once.
    pub changed: Arc<Notify>,
}

impl Engine {
    /// A request found the engine broken (no connection, or one that ended before the response began): its
    /// `/health` is probed until it answers, unless a probe runs already ([`recover`]).
    pub fn found_broken(self: &Arc<Self>) {
        if self.upstream.start_recovery() {
            tokio::spawn(recover(Arc::downgrade(self)));
        }
    }
}
