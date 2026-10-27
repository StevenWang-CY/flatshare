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

use std::sync::{Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use bytes::Bytes;
use flatshare_proto::SloClass;
use http::header::{CONTENT_TYPE, HeaderName, HeaderValue};
use http::uri::PathAndQuery;
use http_body_util::{BodyExt, Limited};
use prometheus_client::metrics::gauge::Gauge;

use crate::config::Canary;
use crate::metrics::EngineState;
use crate::sanitize::{REQUEST_ID, request_id, set_priority};
use crate::server::Shared;
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
    gauges: [Gauge; EngineState::ALL.len()],
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
    pub fn new(canary: Option<Canary>, gauges: [Gauge; EngineState::ALL.len()]) -> Self {
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
            gauges,
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
                tracing::info!("the engine serves again: an inference succeeded");
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
                failures,
                "the engine is degraded: the inference canary failed {failures} times in a row with nothing \
                 succeeding: {cause}"
            );
            self.show(EngineState::Degraded);
        } else {
            tracing::debug!("the inference canary failed: {cause}");
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
        for (state, gauge) in EngineState::ALL.iter().zip(&self.gauges) {
            gauge.set(i64::from(*state == engine));
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Asks the engine for a token whenever nothing succeeded within the canary's interval, until the gateway drains or
/// is gone.
pub(crate) async fn run(shared: Weak<Shared>) {
    loop {
        let Some(wait) = shared.upgrade().and_then(|shared| {
            let canary = shared.health.canary?;
            shared.health.next_probe(&canary, Instant::now())
        }) else {
            return;
        };
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
            // A success meanwhile moves the next canary on.
            continue;
        }
        let Some(shared) = shared.upgrade() else {
            return;
        };
        let Some(canary) = shared.health.canary else {
            return;
        };
        let started = Instant::now();
        shared.health.probed(started);
        let result = probe(&shared, canary.timeout).await;
        shared.metrics.canary(result.is_ok(), started.elapsed());
        match result {
            Ok(()) => shared.health.succeeded(),
            Err(cause) => shared.health.failed(canary.degraded_after, &cause),
        }
    }
}

/// One canary: a one-token completion, read to its end within `timeout`.
async fn probe(shared: &Shared, timeout: Duration) -> Result<(), String> {
    let class = SloClass::Background;
    let signals = shared.upstream_priority;
    let body = match signals.body {
        true => format!("{{{PROBE},\"priority\":{}}}", class.engine_priority()),
        false => format!("{{{PROBE}}}"),
    };
    let uri = shared
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
        let response = shared
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gauges() -> [Gauge; EngineState::ALL.len()] {
        std::array::from_fn(|_| Gauge::default())
    }

    const CANARY_CONFIG: Canary = Canary {
        interval: Duration::from_secs(30),
        timeout: Duration::from_secs(60),
        degraded_after: 2,
    };

    #[test]
    fn the_states_follow_successes_failures_and_the_drain() {
        let shared = gauges();
        let health = Health::new(Some(CANARY_CONFIG), shared.clone());
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
        assert_eq!(Health::new(None, gauges()).state(), EngineState::Serving);
    }

    #[test]
    fn the_canary_asks_only_when_nothing_succeeded_within_its_interval() {
        let health = Health::new(Some(CANARY_CONFIG), gauges());
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
}
