//! The gateway's bounds on the work it holds, whatever its admission policy: inference requests in flight, in all
//! (`max_in_flight`), by class (`[classes.<class>] max_in_flight`) and by tenant (`[tenants.<name>] max_in_flight`), and
//! the bytes of the request bodies it buffers (`max_buffered_body_bytes`).
//!
//! A request takes its permit once its tenant and class are known and before a byte of its body is read, and a
//! refused one is answered at once, so that a flood of requests costs the gateway neither memory nor a wait: 503 with
//! `Retry-After` when the gateway or the class is full, 429 when the tenant is. A body's bytes are taken from the budget
//! chunk by chunk as they arrive, its whole declared length first when it has one, and a body the budget cannot hold
//! gets 503. A request holds its permit and its bytes until its end, whatever ends it: the permit is the request's
//! [`crate::exchange::Exchange`]'s, dropped with its record.
//!
//! While a reload's predecessor drains, the successor counts the predecessor's requests and bytes with its own
//! ([`crate::ledger`]), so that the bounds hold for the one engine both serve; and a gateway that has handed its
//! sockets over admits nothing more.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use flatshare_proto::SloClass;
use prometheus_client::metrics::gauge::Gauge;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use crate::auth::{Tenant, Tenants};

/// What the requests past an intake hold: how many are in flight, in all, by class (strictest first) and by tenant,
/// and the bytes of their bodies. A reload's successor counts its predecessor's with its own.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Held {
    pub in_flight: usize,
    pub classes: [usize; 3],
    pub tenants: BTreeMap<String, usize>,
    pub body_bytes: usize,
}

/// Why the intake refused a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The gateway handed its sockets to a successor and admits nothing more (503): send the request again.
    Draining,
    /// `max_in_flight` requests are in flight (503).
    InFlight,
    /// The class's `max_in_flight` requests are (503).
    Class,
    /// The tenant's `max_in_flight` requests are (429).
    Tenant,
    /// The bodies held would pass `max_buffered_body_bytes` (503).
    BodyBytes,
}

/// The intake's bounds, from the configuration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub in_flight: usize,
    /// By class, strictest first.
    pub classes: [usize; 3],
    pub body_bytes: usize,
}

/// The intake's gauges: its own requests in flight by class and their bodies' bytes, and the predecessor's requests
/// it counts during a reload.
#[derive(Clone, Debug)]
pub(crate) struct Gauges {
    pub in_flight: [Gauge; 3],
    pub body_bytes: Gauge,
    pub predecessor_in_flight: Gauge,
}

#[derive(Debug)]
pub(crate) struct Intake {
    limits: Limits,
    /// Each tenant's name and bound, by its index.
    tenants: Vec<(Arc<str>, usize)>,
    state: Mutex<State>,
    /// Set when the gateway hands its sockets to a successor: it admits nothing more.
    closed: AtomicBool,
    /// Told of every change to what it holds, for a successor that counts it ([`crate::ledger`]).
    changed: Arc<Notify>,
    gauges: Gauges,
}

/// What the intake's own requests hold, and what it counts of a predecessor's.
#[derive(Debug, Default)]
struct State {
    in_flight: usize,
    classes: [usize; 3],
    /// By tenant index.
    tenants: Vec<usize>,
    body_bytes: usize,
    predecessor: Held,
}

/// A request's place in the intake, from its headers to its end, with the bytes of its body it took.
#[derive(Debug)]
pub(crate) struct Permit {
    intake: Arc<Intake>,
    tenant: usize,
    class: SloClass,
    /// Its body's bytes.
    pub body: BodyBytes,
}

/// The bytes a body takes from the gateway's budget as it is read, given back when dropped: a request's, in its
/// [`Permit`], or a batch of lifecycle events'.
#[derive(Debug)]
pub(crate) struct BodyBytes {
    intake: Arc<Intake>,
    taken: usize,
}

impl Intake {
    pub fn new(limits: Limits, tenants: &Tenants, changed: Arc<Notify>, gauges: Gauges) -> Self {
        let tenants: Vec<_> = tenants
            .all()
            .iter()
            .map(|tenant| (Arc::clone(&tenant.name), tenant.max_in_flight))
            .collect();
        let state = State {
            tenants: vec![0; tenants.len()],
            ..State::default()
        };
        Self {
            limits,
            tenants,
            state: Mutex::new(state),
            closed: AtomicBool::new(false),
            changed,
            gauges,
        }
    }

    /// Takes a place for a request of `tenant` and `class`, or says which bound is full. It never waits.
    pub fn admit(self: &Arc<Self>, tenant: &Tenant, class: SloClass) -> Result<Permit, Refused> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Refused::Draining);
        }
        let c = class as usize;
        let mut state = self.lock();
        let predecessor = &state.predecessor;
        let tenant_held = state.tenants[tenant.index]
            + predecessor.tenants.get(&*tenant.name).copied().unwrap_or(0);
        // The tenant's own bound first: a tenant over its share is told so (429) even when the gateway is full too.
        if tenant_held >= tenant.max_in_flight {
            return Err(Refused::Tenant);
        }
        if state.classes[c] + predecessor.classes[c] >= self.limits.classes[c] {
            return Err(Refused::Class);
        }
        if state.in_flight + predecessor.in_flight >= self.limits.in_flight {
            return Err(Refused::InFlight);
        }
        state.in_flight += 1;
        state.classes[c] += 1;
        state.tenants[tenant.index] += 1;
        drop(state);
        self.gauges.in_flight[c].inc();
        self.changed.notify_waiters();
        Ok(Permit {
            intake: Arc::clone(self),
            tenant: tenant.index,
            class,
            body: self.body(),
        })
    }

    /// A reservation of body bytes that holds none yet.
    pub fn body(self: &Arc<Self>) -> BodyBytes {
        BodyBytes {
            intake: Arc::clone(self),
            taken: 0,
        }
    }

    /// What the intake's requests hold, with what it counts of a predecessor's: what a successor is to count.
    pub fn held(&self) -> Held {
        let state = self.lock();
        let mut held = state.predecessor.clone();
        held.in_flight += state.in_flight;
        held.body_bytes += state.body_bytes;
        for (counted, own) in held.classes.iter_mut().zip(state.classes) {
            *counted += own;
        }
        for ((name, _), &own) in self.tenants.iter().zip(&state.tenants) {
            if own > 0 {
                *held.tenants.entry(name.to_string()).or_default() += own;
            }
        }
        held
    }

    /// What a predecessor holds now, counted with the intake's own until it exits.
    pub fn set_predecessor(&self, held: Held) {
        self.gauges
            .predecessor_in_flight
            .set(i64::try_from(held.in_flight).unwrap_or(i64::MAX));
        self.lock().predecessor = held;
        self.changed.notify_waiters();
    }

    /// Admits nothing more: the gateway handed its sockets to a successor, which admits every request from now on.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.changed.notify_waiters();
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Why a request was refused, for its client.
    pub fn message(&self, refused: Refused, tenant: &Tenant, class: SloClass) -> String {
        let limits = &self.limits;
        match refused {
            Refused::Draining => {
                "the gateway handed its sockets to a successor and admits nothing more; send the request again"
                    .into()
            }
            Refused::InFlight => format!(
                "the gateway has {} requests in flight (max_in_flight); send the request again",
                limits.in_flight
            ),
            Refused::Class => format!(
                "the gateway has {} {class} requests in flight (classes.{class}.max_in_flight); send the request again",
                limits.classes[class as usize]
            ),
            Refused::Tenant => format!(
                "tenant {} has {} requests in flight (tenants.{}.max_in_flight)",
                tenant.name, tenant.max_in_flight, tenant.name
            ),
            Refused::BodyBytes => format!(
                "the gateway holds {} bytes of request bodies (max_buffered_body_bytes); send the request again",
                limits.body_bytes
            ),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl BodyBytes {
    /// Takes `bytes` more of the budget, or refuses them when the bodies held, a predecessor's counted, would pass
    /// it.
    pub fn take(&mut self, bytes: usize) -> Result<(), Refused> {
        if bytes == 0 {
            return Ok(());
        }
        let intake = &self.intake;
        let mut state = intake.lock();
        let held = state.body_bytes + state.predecessor.body_bytes;
        if held.saturating_add(bytes) > intake.limits.body_bytes {
            return Err(Refused::BodyBytes);
        }
        state.body_bytes += bytes;
        drop(state);
        self.taken += bytes;
        intake
            .gauges
            .body_bytes
            .inc_by(i64::try_from(bytes).unwrap_or(i64::MAX));
        intake.changed.notify_waiters();
        Ok(())
    }

    /// The bytes taken so far.
    pub fn taken(&self) -> usize {
        self.taken
    }
}

impl Drop for BodyBytes {
    fn drop(&mut self) {
        if self.taken == 0 {
            return;
        }
        let intake = &self.intake;
        intake.lock().body_bytes -= self.taken;
        intake
            .gauges
            .body_bytes
            .dec_by(i64::try_from(self.taken).unwrap_or(i64::MAX));
        intake.changed.notify_waiters();
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        let intake = &self.intake;
        let c = self.class as usize;
        let mut state = intake.lock();
        state.in_flight -= 1;
        state.classes[c] -= 1;
        state.tenants[self.tenant] -= 1;
        drop(state);
        intake.gauges.in_flight[c].dec();
        intake.changed.notify_waiters();
    }
}

/// The tests: each bound refusing at once and a permit freeing its place; one budget for every body, events' among
/// them; a predecessor's requests counted, by tenant too, until it is gone; a closed intake admitting nothing.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{self, CacheSaltSecret};

    fn intake(limits: Limits, tenants: &[(&str, usize)]) -> (Arc<Intake>, Tenants) {
        let tenants: Vec<config::TenantConfig> = tenants
            .iter()
            .map(|&(name, max_in_flight)| config::TenantConfig {
                name: name.into(),
                keys_sha256: Vec::new(),
                trusted: false,
                default_class: SloClass::Background,
                max_class: SloClass::Realtime,
                max_in_flight,
            })
            .collect();
        let tenants = Tenants::new(&tenants, &CacheSaltSecret([0; 32]));
        let gauges = Gauges {
            in_flight: std::array::from_fn(|_| Gauge::default()),
            body_bytes: Gauge::default(),
            predecessor_in_flight: Gauge::default(),
        };
        let intake = Intake::new(limits, &tenants, Arc::new(Notify::new()), gauges);
        (Arc::new(intake), tenants)
    }

    const LIMITS: Limits = Limits {
        in_flight: 4,
        classes: [2, 4, 3],
        body_bytes: 100,
    };

    #[test]
    fn each_bound_refuses_at_once_and_a_permit_frees_its_place() {
        use SloClass::*;
        let (intake, tenants) = intake(LIMITS, &[("a", 2), ("b", 4)]);
        let [a, b] = [&tenants.all()[0], &tenants.all()[1]];
        let first = intake.admit(a, Background).unwrap();
        let _second = intake.admit(a, Background).unwrap();
        // Tenant a holds its two; b may go on, until the class holds its three.
        assert_eq!(intake.admit(a, Realtime).unwrap_err(), Refused::Tenant);
        let _third = intake.admit(b, Background).unwrap();
        assert_eq!(intake.admit(b, Background).unwrap_err(), Refused::Class);
        // The gateway holds four at most.
        let _fourth = intake.admit(b, Realtime).unwrap();
        assert_eq!(intake.admit(b, Interactive).unwrap_err(), Refused::InFlight);
        drop(first);
        let _again = intake.admit(a, Background).unwrap();
        let held = intake.held();
        assert_eq!((held.in_flight, held.classes), (4, [1, 0, 3]));
        assert_eq!(held.tenants.get("a"), Some(&2));
    }

    #[test]
    fn bodies_take_their_bytes_from_one_budget() {
        let (intake, tenants) = intake(LIMITS, &[("a", 4)]);
        let a = &tenants.all()[0];
        let mut one = intake.admit(a, SloClass::Background).unwrap();
        let mut two = intake.admit(a, SloClass::Background).unwrap();
        one.body.take(60).unwrap();
        assert_eq!(two.body.take(41).unwrap_err(), Refused::BodyBytes);
        two.body.take(40).unwrap();
        assert_eq!(intake.held().body_bytes, 100);
        drop(one);
        two.body.take(60).unwrap();
        assert_eq!((two.body.taken(), intake.held().body_bytes), (100, 100));
        // A batch of events takes from the same budget, without a place among the requests.
        let mut events = intake.body();
        assert_eq!(events.take(1).unwrap_err(), Refused::BodyBytes);
        drop(two);
        events.take(100).unwrap();
        assert_eq!(
            intake.held(),
            Held {
                body_bytes: 100,
                ..Held::default()
            }
        );
    }

    #[test]
    fn a_predecessors_requests_count_until_it_is_gone() {
        let (intake, tenants) = intake(LIMITS, &[("a", 3)]);
        let a = &tenants.all()[0];
        let predecessor = Held {
            in_flight: 3,
            classes: [0, 0, 3],
            tenants: BTreeMap::from([("a".to_owned(), 2), ("gone".to_owned(), 1)]),
            body_bytes: 90,
        };
        intake.set_predecessor(predecessor.clone());
        assert_eq!(
            intake.admit(a, SloClass::Background).unwrap_err(),
            Refused::Class
        );
        let mut realtime = intake.admit(a, SloClass::Realtime).unwrap();
        // The tenant's two there and one here; the gateway's three there and one here.
        assert_eq!(
            intake.admit(a, SloClass::Interactive).unwrap_err(),
            Refused::Tenant
        );
        assert_eq!(realtime.body.take(11).unwrap_err(), Refused::BodyBytes);
        realtime.body.take(10).unwrap();
        // A successor of this gateway counts both.
        let held = intake.held();
        assert_eq!(
            (held.in_flight, held.classes, held.body_bytes),
            (4, [1, 0, 3], 100)
        );
        assert_eq!(held.tenants.get("a"), Some(&3));
        intake.set_predecessor(Held::default());
        assert!(intake.admit(a, SloClass::Background).is_ok());
        assert!(intake.admit(a, SloClass::Background).is_ok());
    }

    #[test]
    fn a_closed_intake_admits_nothing() {
        let (intake, tenants) = intake(LIMITS, &[("a", 3)]);
        intake.close();
        assert!(intake.is_closed());
        assert_eq!(
            intake
                .admit(&tenants.all()[0], SloClass::Realtime)
                .unwrap_err(),
            Refused::Draining
        );
    }
}
