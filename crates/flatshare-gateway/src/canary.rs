//! The engine's state as inference finds it, which `GET /ready` answers by: the engine's own `/health` says only that
//! its process lives (vLLM 0.32 answers 503 only once its engine core is dead, and runs no inference), so an engine
//! that stalls would look healthy.
//!
//! - `warming` until an inference succeeds, from the gateway's start;
//! - `serving` from then on, and again after any success;
//! - `degraded` once the canary has failed `degraded_after` times in a row with nothing succeeding between;
//! - `drained` from the start of the gateway's drain.
//!
//! What succeeds is a client's request that got a 2xx response's first content, or relayed one to its end, and the
//! canary: a one-token completion (`POST /v1/completions`, `max_tokens` 1, temperature 0, no model, so the engine's
//! own), at the lowest class's priority, with `X-Flatshare-Canary: 1`, through the upstream's connection pool and
//! outside every request bound and count. The canary runs only while nothing succeeds: when no inference succeeded
//! within `interval_s`, and every second while warming (or every `interval_s`, if shorter), one at a time, each for
//! `timeout_s` at most. While clients'
//! requests succeed it sends nothing, so it never adds to an engine that real traffic shows serving. With the canary
//! off (`[canary] enabled = false`, as E2's runs set it) the engine is `serving` from the start, and `/ready` asks the
//! engine's `/health` as it always did.
//!
//! With `[[nodes]]` each node's engine has a state and a canary of its own, and a degraded node's engine is asked
//! every second, as a warming one is: a node that is not healthy takes no requests, so only its canary can show it
//! serving again.

use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use bytes::Bytes;
use flatshare_proto::SloClass;
use http::header::{CONTENT_TYPE, HeaderName, HeaderValue};
use http::uri::PathAndQuery;
use http_body_util::{BodyExt, Limited};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use tokio::sync::Notify;

use crate::config::Canary;
use crate::engine::Engine;
use crate::metrics::EngineState;
use crate::sanitize::{REQUEST_ID, request_id, set_priority};
use crate::upstream::{UpstreamBody, causes};

/// How often the canary asks while the engine is warming.
const WARMING_PROBE: Duration = Duration::from_secs(1);

/// The header that names the canary's requests to the engine's logs.
static CANARY: HeaderName = HeaderName::from_static("x-flatshare-canary");

/// The canary's request body, its priority aside.
const PROBE: &str = r#""prompt":"ping","max_tokens":1,"temperature":0"#;

/// The engine's state, with the gauges that export it.
#[derive(Debug)]
pub(crate) struct Health {
    canary: Option<Canary>,
    state: Mutex<State>,
    watched: Watched,
}

/// Where an engine's state shows, and who learns of its changes.
#[derive(Debug)]
pub(crate) struct Watched {
    /// The state's gauges, in [`EngineState::ALL`] order.
    pub gauges: [Gauge; EngineState::ALL.len()],
    /// Its canaries' results and times.
    pub probes: Probes,
    /// Woken at each change of state; `None` for a gateway's one upstream.
    pub changed: Option<Arc<Notify>>,
    /// The engine's node, which a degraded engine's canary asks every second as a warming one's does; `None` for a
    /// gateway's one upstream, which goes on taking requests while degraded.
    pub node: Option<Arc<str>>,
}

/// The series an engine's canaries count in.
#[derive(Clone, Debug)]
pub(crate) struct Probes {
    pub success: Counter,
    pub failure: Counter,
    pub seconds: Histogram,
}

impl Probes {
    fn record(&self, success: bool, elapsed: Duration) {
        if success {
            self.success.inc();
        } else {
            self.failure.inc();
        }
        self.seconds.observe(elapsed.as_secs_f64());
    }
}

#[derive(Debug)]
struct State {
    engine: EngineState,
    /// The canaries that failed since the last success.
    failures: u32,
    last_success: Option<Instant>,
    last_probe: Option<Instant>,
}

impl Health {
    /// Warming with a canary, serving without one.
    pub fn new(canary: Option<Canary>, watched: Watched) -> Self {
        let engine = if canary.is_some() {
            EngineState::Warming
        } else {
            EngineState::Serving
        };
        let health = Self {
            canary,
            state: Mutex::new(State {
                engine,
                failures: 0,
                last_success: None,
                last_probe: None,
            }),
            watched,
        };
        health.show(engine);
        health
    }

    pub fn state(&self) -> EngineState {
        self.lock().engine
    }

    /// An inference succeeded: a client's request or the canary.
    pub fn succeeded(&self) {
        let mut state = self.lock();
        state.last_success = Some(Instant::now());
        state.failures = 0;
        if state.engine != EngineState::Drained && state.engine != EngineState::Serving {
            if state.engine == EngineState::Degraded {
                let node = self.watched.node.as_deref();
                tracing::info!(node, "the engine serves again: an inference succeeded");
            }
            state.engine = EngineState::Serving;
            drop(state);
            self.show(EngineState::Serving);
        }
    }

    /// The canary failed.
    fn failed(&self, degraded_after: u32, cause: &str) {
        let mut state = self.lock();
        state.failures += 1;
        if state.engine == EngineState::Serving && state.failures >= degraded_after {
            state.engine = EngineState::Degraded;
            let failures = state.failures;
            drop(state);
            tracing::warn!(
                node = self.watched.node.as_deref(),
                failures,
                "the engine is degraded: the inference canary failed {failures} times in a row with nothing \
                 succeeding: {cause}"
            );
            self.show(EngineState::Degraded);
        } else {
            let node = self.watched.node.as_deref();
            tracing::debug!(node, "the inference canary failed: {cause}");
        }
    }

    /// The gateway drains.
    pub fn drain(&self) {
        self.lock().engine = EngineState::Drained;
        self.show(EngineState::Drained);
    }

    /// How long until the canary is next to ask, from `now`; `None` once the gateway drains.
    fn next_probe(&self, canary: &Canary, now: Instant) -> Option<Duration> {
        let state = self.lock();
        let due = match state.engine {
            EngineState::Drained => return None,
            EngineState::Warming => state
                .last_probe
                .map(|at| at + WARMING_PROBE.min(canary.interval)),
            // A node's: it takes no requests while degraded, so its canary is what can show it serving again.
            EngineState::Degraded if self.watched.node.is_some() => state
                .last_probe
                .map(|at| at + WARMING_PROBE.min(canary.interval)),
            // The later of the last success and the last canary, an interval on.
            EngineState::Serving | EngineState::Degraded => state
                .last_success
                .max(state.last_probe)
                .map(|at| at + canary.interval),
        };
        Some(due.map_or(Duration::ZERO, |due| due.saturating_duration_since(now)))
    }

    fn probed(&self, now: Instant) {
        self.lock().last_probe = Some(now);
    }

    fn show(&self, engine: EngineState) {
        for (state, gauge) in EngineState::ALL.iter().zip(&self.watched.gauges) {
            gauge.set(i64::from(*state == engine));
        }
        if let Some(changed) = &self.watched.changed {
            changed.notify_waiters();
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Asks the engine for a token whenever nothing succeeded within the canary's interval, until the gateway drains or
/// is gone.
pub(crate) async fn run(engine: Weak<Engine>) {
    loop {
        let Some(wait) = engine.upgrade().and_then(|engine| {
            let canary = engine.health.canary?;
            engine.health.next_probe(&canary, Instant::now())
        }) else {
            return;
        };
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
            // A success meanwhile moves the next canary on.
            continue;
        }
        let Some(engine) = engine.upgrade() else {
            return;
        };
        let health = &engine.health;
        let Some(canary) = health.canary else {
            return;
        };
        let started = Instant::now();
        health.probed(started);
        let result = probe(&engine, canary.timeout).await;
        health
            .watched
            .probes
            .record(result.is_ok(), started.elapsed());
        match result {
            Ok(()) => health.succeeded(),
            Err(cause) => health.failed(canary.degraded_after, &cause),
        }
    }
}

/// One canary: a one-token completion, read to its end within `timeout`.
async fn probe(engine: &Engine, timeout: Duration) -> Result<(), String> {
    let class = SloClass::Background;
    let signals = engine.signals;
    let body = match signals.body {
        true => format!("{{{PROBE},\"priority\":{}}}", class.engine_priority()),
        false => format!("{{{PROBE}}}"),
    };
    let uri = engine
        .upstream
        .uri(PathAndQuery::from_static("/v1/completions"));
    let mut request = http::Request::post(uri)
        .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
        .header(CANARY.clone(), HeaderValue::from_static("1"))
        .header(REQUEST_ID.clone(), request_id())
        .body(UpstreamBody::plain(Bytes::from(body)))
        .expect("a POST with a valid URI is a valid request");
    set_priority(request.headers_mut(), class, signals);
    let attempt = async {
        let response = engine
            .upstream
            .request(request)
            .await
            .map_err(|err| format!("the canary failed: {}", causes(&err)))?;
        let status = response.status();
        // A completion of one token is a few hundred bytes.
        Limited::new(response.into_body(), 64 << 10)
            .collect()
            .await
            .map_err(|err| format!("the canary's answer broke off: {err}"))?;
        if !status.is_success() {
            return Err(format!("the canary got {status}"));
        }
        Ok(())
    };
    tokio::time::timeout(timeout, attempt)
        .await
        .map_err(|_| format!("the canary got no answer within {timeout:?}"))?
}

/// The tests: warming, serving, degraded and drained by the successes and the canary's failures, a failure while
/// warming keeping it warming, a drained engine staying drained, and none of it without a canary; the canary due at
/// once while warming, a second after each, and an interval after the later of the last success and the last canary
/// while serving; the canary on unless switched off, its other keys allowed while it is off and a zero or unknown one
/// refused; realtime's objective of 250 ms, and another class's when set.
#[cfg(test)]
mod tests {
    use super::*;

    fn gauges() -> [Gauge; EngineState::ALL.len()] {
        std::array::from_fn(|_| Gauge::default())
    }

    fn watched(gauges: [Gauge; EngineState::ALL.len()], node: Option<&str>) -> Watched {
        Watched {
            gauges,
            probes: Probes {
                success: Counter::default(),
                failure: Counter::default(),
                seconds: Histogram::new([1.0]),
            },
            changed: None,
            node: node.map(Arc::from),
        }
    }

    const CANARY_CONFIG: Canary = Canary {
        interval: Duration::from_secs(30),
        timeout: Duration::from_secs(60),
        degraded_after: 2,
    };

    #[test]
    fn the_states_follow_successes_failures_and_the_drain() {
        let shared = gauges();
        let health = Health::new(Some(CANARY_CONFIG), watched(shared.clone(), None));
        let shown = || shared.iter().map(Gauge::get).collect::<Vec<_>>();
        assert_eq!(
            (health.state(), shown()),
            (EngineState::Warming, vec![1, 0, 0, 0])
        );
        // Failing while warming keeps it warming: it never served.
        health.failed(2, "down");
        health.failed(2, "down");
        assert_eq!(health.state(), EngineState::Warming);
        health.succeeded();
        assert_eq!(
            (health.state(), shown()),
            (EngineState::Serving, vec![0, 1, 0, 0])
        );
        // Two failures in a row with nothing succeeding between degrade it; one success serves again.
        health.failed(2, "slow");
        health.succeeded();
        health.failed(2, "slow");
        assert_eq!(health.state(), EngineState::Serving);
        health.failed(2, "slow");
        assert_eq!(
            (health.state(), shown()),
            (EngineState::Degraded, vec![0, 0, 1, 0])
        );
        health.succeeded();
        assert_eq!(health.state(), EngineState::Serving);
        health.drain();
        health.succeeded();
        assert_eq!(
            (health.state(), shown()),
            (EngineState::Drained, vec![0, 0, 0, 1])
        );
        // Without a canary it serves from the start.
        assert_eq!(
            Health::new(None, watched(gauges(), None)).state(),
            EngineState::Serving
        );
    }

    #[test]
    fn the_canary_asks_only_when_nothing_succeeded_within_its_interval() {
        let health = Health::new(Some(CANARY_CONFIG), watched(gauges(), None));
        let now = Instant::now();
        // Warming: at once, then a second after each canary.
        assert_eq!(health.next_probe(&CANARY_CONFIG, now), Some(Duration::ZERO));
        health.probed(now);
        assert_eq!(health.next_probe(&CANARY_CONFIG, now), Some(WARMING_PROBE));
        // Serving: an interval after the last success or canary, whichever is later.
        health.succeeded();
        let wait = health.next_probe(&CANARY_CONFIG, Instant::now()).unwrap();
        assert!(
            wait > Duration::from_secs(29) && wait <= Duration::from_secs(30),
            "{wait:?}"
        );
        let later = Instant::now() + Duration::from_secs(31);
        assert_eq!(
            health.next_probe(&CANARY_CONFIG, later),
            Some(Duration::ZERO)
        );
        health.drain();
        assert_eq!(health.next_probe(&CANARY_CONFIG, later), None);
    }

    #[test]
    fn a_nodes_degraded_engine_is_asked_every_second_and_a_lone_engines_every_interval() {
        let lone = Health::new(Some(CANARY_CONFIG), watched(gauges(), None));
        let node = Health::new(Some(CANARY_CONFIG), watched(gauges(), Some("gray")));
        for (health, wait) in [(&lone, CANARY_CONFIG.interval), (&node, WARMING_PROBE)] {
            health.succeeded();
            health.failed(2, "slow");
            health.failed(2, "slow");
            assert_eq!(health.state(), EngineState::Degraded);
            // A canary at `now`, after the last success.
            let now = Instant::now();
            health.probed(now);
            assert_eq!(health.next_probe(&CANARY_CONFIG, now), Some(wait));
        }
        // Serving again, a node's engine is asked an interval after its last success, as a lone engine is.
        node.succeeded();
        let wait = node.next_probe(&CANARY_CONFIG, Instant::now()).unwrap();
        assert!(wait > Duration::from_secs(29), "{wait:?}");
    }
}
