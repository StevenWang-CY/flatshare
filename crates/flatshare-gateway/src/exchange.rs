//! One record per API request, from receipt to its end: the request id, what the gateway resolved, and how the
//! request ended. Dropping the record writes the request's metrics and its log line, so every exit path is
//! recorded exactly once, including a client that leaves while its handler waits (hyper then drops the
//! handler's future, and with it the record).
//!
//! An inference request is offered once, as its class is known (or at its end, if it ended first), admitted once if
//! it went upstream, and ended once, with one outcome: so offered minus ended is what is in flight, a request that
//! hangs included, and every failure is counted where the service level reads it. The SLI counts a request as valid
//! unless its client is at fault (an [`Outcome::clients_fault`]) or its client cancelled it before the class's
//! first-content objective had passed without its first content, and as good when it completed, a 2xx response
//! relayed to its end, or was cancelled by its client after its first content, with its first content within the
//! objective (any time, for a class without one; a class without one counts no cancellation).

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::{IntoResponse, Response};
use flatshare_core::admission::Deferral;
use flatshare_proto::{SessionId, SloClass};
use http::StatusCode;
use http::header::{CONTENT_TYPE, HeaderMap, HeaderValue, RETRY_AFTER};
use tracing::Level;

use crate::bounds;
use crate::canary::Health;
use crate::deadline::{Deadlines, Phase, Timeout};
use crate::harness::Refusal;
use crate::intake::{Permit, Refused};
use crate::metrics::{Metrics, Outcome, Route, UpstreamError};
use crate::nodes::{Placed, Unplaced};
use crate::sanitize::{REQUEST_ID, RESOLVED_SESSION_ID, request_id, session_header};
use crate::usage::Summary;

/// The status recorded for a request whose client left before any response: nginx's "client closed request".
const CLIENT_CLOSED: u16 = 499;

#[derive(Debug)]
pub(crate) struct Exchange {
    metrics: Arc<Metrics>,
    /// Sent upstream as `X-Request-Id` and returned to the client.
    pub id: HeaderValue,
    pub route: Route,
    pub received: Instant,
    pub tenant: Option<Arc<str>>,
    pub class: Option<SloClass>,
    pub session: Option<SessionId>,
    /// The status sent to the client; `None` until a response exists.
    pub status: Option<StatusCode>,
    /// Whether the request was sent upstream, which makes it count towards outcomes.
    pub forwarded: bool,
    /// How long the request waited at the admission point; `None` without one.
    pub admission_wait: Option<Duration>,
    /// Why the lifecycle policy deferred it.
    pub deferred: Option<Deferral>,
    /// Whether the lifecycle policy admitted it regardless, after `max_defer_s`.
    pub aged: bool,
    /// Whether the lifecycle policy's degrade capped its reply.
    pub degraded: bool,
    /// How many times it was sent upstream again after a preemption.
    pub resubmits: u32,
    pub first_byte: Option<Duration>,
    pub summary: Summary,
    /// The request's place in the intake and its body's bytes, given back when the record ends.
    pub permit: Option<Permit>,
    /// Its deadlines, once its class is known.
    pub deadlines: Option<Deadlines>,
    /// Its class's first-content objective, for the SLI.
    pub objective: Option<Duration>,
    /// The engine's state, which an inference request's success keeps serving.
    pub health: Option<Arc<Health>>,
    /// With `[[nodes]]`: the request's node, for its log line, and its place there, given back when the record
    /// ends.
    pub node: Option<Arc<str>>,
    pub placed: Option<Placed>,
    /// How many times it was sent again on another node after its node failed before its first content.
    pub retries: u32,
    /// Whether it was counted as offered.
    offered: bool,
    outcome: Option<Outcome>,
    error: Option<String>,
}

impl Exchange {
    pub fn new(metrics: &Arc<Metrics>, route: Route) -> Self {
        Self {
            metrics: Arc::clone(metrics),
            id: request_id(),
            route,
            received: Instant::now(),
            tenant: None,
            class: None,
            session: None,
            status: None,
            forwarded: false,
            admission_wait: None,
            deferred: None,
            aged: false,
            degraded: false,
            resubmits: 0,
            first_byte: None,
            summary: Summary::default(),
            permit: None,
            deadlines: None,
            objective: None,
            health: None,
            node: None,
            placed: None,
            retries: 0,
            offered: false,
            outcome: None,
            error: None,
        }
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Counts an inference request as offered, once, with its tenant, route and class as they are known.
    pub fn offer(&mut self) {
        if self.route.is_inference() && !std::mem::replace(&mut self.offered, true) {
            self.metrics
                .offered(self.tenant.as_ref(), self.route, self.class);
        }
    }

    /// Counts an inference request as admitted: past the intake, the admission point and the readiness check, on its
    /// way upstream; `policy` when an admission policy decided when.
    pub fn admit(&self, policy: bool) {
        if self.route.is_inference() {
            self.metrics
                .admitted(self.tenant.as_ref(), self.route, self.class, policy);
        }
    }

    /// An inference succeeded: a 2xx response's first content arrived, or it was relayed to its end.
    pub fn succeeded(&self) {
        if let Some(health) = &self.health {
            health.succeeded();
        }
    }

    /// Adds the request id and, once resolved, the session id to response headers.
    pub fn decorate(&self, headers: &mut HeaderMap) {
        headers.insert(REQUEST_ID.clone(), self.id.clone());
        if let Some(session) = &self.session {
            headers.insert(RESOLVED_SESSION_ID.clone(), session_header(session));
        }
    }

    /// Answers with a JSON body generated by the gateway.
    pub fn respond(&mut self, status: StatusCode, body: &serde_json::Value) -> Response {
        let content_type = [(CONTENT_TYPE, HeaderValue::from_static("application/json"))];
        let mut response = (status, content_type, body.to_string()).into_response();
        self.decorate(response.headers_mut());
        self.status = Some(status);
        response
    }

    /// Answers with an OpenAI-format error, `{"error": {"message", "type", "code"}}`, and records it.
    pub fn error(mut self, status: StatusCode, code: &str, message: impl Into<String>) -> Response {
        let message = message.into();
        let kind = match status {
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::FORBIDDEN => "permission_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            s if s.is_server_error() => "server_error",
            _ => "invalid_request_error",
        };
        let body = serde_json::json!({"error": {"message": message, "type": kind, "code": code}});
        if self.forwarded {
            self.outcome.get_or_insert(Outcome::TransportError);
        } else if status.is_client_error() {
            self.outcome.get_or_insert(Outcome::Invalid);
        }
        self.error = Some(message);
        self.respond(status, &body)
    }

    /// The 503 for a request that no ready engine can take, sent without going upstream. `Retry-After` is `retry`,
    /// the interval of the readiness probe that will say when an engine answers again, in whole seconds.
    pub fn not_ready(
        mut self,
        code: &str,
        message: impl Into<String>,
        retry: Duration,
    ) -> Response {
        self.outcome.get_or_insert(Outcome::NotReady);
        let mut response = self.error(StatusCode::SERVICE_UNAVAILABLE, code, message);
        let seconds = retry.as_secs_f64().ceil().max(1.0) as u64;
        response
            .headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from(seconds));
        response
    }

    /// The 400 for a body whose reply limits or choices the gateway does not forward ([`crate::bounds`]), with the
    /// field at fault as the error's `param`.
    pub fn invalid_field(mut self, invalid: &bounds::Invalid) -> Response {
        let message = invalid.to_string();
        let body = serde_json::json!({"error": {
            "message": message, "type": "invalid_request_error", "param": invalid.field(), "code": invalid.code(),
        }});
        self.outcome = Some(Outcome::Invalid);
        self.error = Some(message);
        self.respond(StatusCode::BAD_REQUEST, &body)
    }

    /// An error with `Retry-After: 1`, for a request the gateway did not take that its client may send again.
    fn retry(self, status: StatusCode, code: &str, message: impl Into<String>) -> Response {
        let mut response = self.error(status, code, message);
        response
            .headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from_static("1"));
        response
    }

    /// The answer to a request the intake refused, before its body was read or as its bytes passed the budget: 429
    /// when its tenant has its requests in flight, 503 when the gateway, its class or the bodies' budget is full, or
    /// when the gateway admits nothing more; each with `Retry-After: 1`.
    pub fn intake_refused(mut self, refused: Refused, message: String) -> Response {
        let (status, code) = match refused {
            Refused::Draining => (StatusCode::SERVICE_UNAVAILABLE, "draining"),
            Refused::InFlight => (StatusCode::SERVICE_UNAVAILABLE, "in_flight_limit"),
            Refused::Class => (StatusCode::SERVICE_UNAVAILABLE, "class_in_flight_limit"),
            Refused::Tenant => (StatusCode::TOO_MANY_REQUESTS, "tenant_in_flight_limit"),
            Refused::BodyBytes => (StatusCode::SERVICE_UNAVAILABLE, "body_budget_exhausted"),
        };
        self.outcome = Some(match refused {
            Refused::Draining => Outcome::Drained,
            Refused::InFlight => Outcome::RefusedInFlight,
            Refused::Class => Outcome::RefusedClass,
            Refused::Tenant => Outcome::RefusedTenant,
            Refused::BodyBytes => Outcome::RefusedBodyBytes,
        });
        self.retry(status, code, message)
    }

    /// The answer to a request whose `timeout` passed before any response began, by the `phase` it was in: 408 for an
    /// upload, 503 with `Retry-After: 1` for a wait at the admission point (the engine never saw the request, or saw
    /// only an attempt that was dropped), 504 for the upstream.
    pub fn timed_out(mut self, phase: Phase, timeout: Timeout) -> Response {
        let message = self.deadlines.map_or_else(
            || format!("the request passed its {}", timeout.key()),
            |deadlines| deadlines.message(timeout),
        );
        self.outcome = Some(Outcome::timeout(timeout));
        match phase {
            Phase::Upload => self.error(StatusCode::REQUEST_TIMEOUT, "upload_timeout", message),
            Phase::Queue => self.retry(StatusCode::SERVICE_UNAVAILABLE, "queue_timeout", message),
            Phase::Upstream => {
                self.metrics.upstream_error(UpstreamError::Timeout);
                self.error(StatusCode::GATEWAY_TIMEOUT, "upstream_timeout", message)
            }
        }
    }

    /// The 503 for a session that no node takes, with `Retry-After: 1`: no available node of its class's role, every
    /// available realtime node at its `max_calls` or background node at its `max_agents`, or (`POST /v1/sessions`) a
    /// table whose every session is busy.
    pub fn unplaced(self, unplaced: Unplaced, class: SloClass) -> Response {
        let (code, message) = match unplaced {
            Unplaced::NoNode => (
                "no_healthy_node",
                format!("no healthy node takes {class} sessions; ask again"),
            ),
            Unplaced::NoCallSlot => (
                "no_call_slot",
                "every healthy realtime node has max_calls calls: the call does not fit the nodes that serve"
                    .to_owned(),
            ),
            Unplaced::NoAgentSlot => (
                "no_agent_slot",
                "every healthy background node has max_agents agents' sessions: the session does not fit the nodes \
                 that serve"
                    .to_owned(),
            ),
            Unplaced::Full => (
                "placement_full",
                format!(
                    "each of the {} sessions placed has a request in flight",
                    crate::nodes::SESSIONS
                ),
            ),
        };
        self.not_ready(code, message, Duration::from_secs(1))
    }

    /// The upstream broke a response off after it began: with `[[nodes]]`, its node's engine is probed, as for a
    /// connection that fails.
    pub fn broke(&self) {
        if let Some(placed) = &self.placed {
            placed.node.engine.found_broken();
        }
    }

    /// The 401 for a request without a known API key on a listener without a tenant.
    pub fn unauthorized(self) -> Response {
        self.error(
            StatusCode::UNAUTHORIZED,
            "invalid_api_key",
            "missing or unknown API key",
        )
    }

    /// The error for a request the admission point refused: 429 when its class's queue was full, 503 when it
    /// was not admitted in time.
    pub fn refused(mut self, refusal: Refusal) -> Response {
        let class = self.class.map_or("", SloClass::as_str);
        self.outcome = Some(match refusal {
            Refusal::QueueFull | Refusal::Queued => Outcome::RefusedQueue,
            Refusal::Timeout => Outcome::TimeoutQueue,
            Refusal::Draining => Outcome::Drained,
        });
        match refusal {
            Refusal::QueueFull => self.retry(
                StatusCode::TOO_MANY_REQUESTS,
                "queue_full",
                format!("the admission queue of {class} requests is full"),
            ),
            Refusal::Queued => self.retry(
                StatusCode::TOO_MANY_REQUESTS,
                "queue_full",
                "the admission point holds max_queued requests waiting",
            ),
            Refusal::Timeout => self.retry(
                StatusCode::SERVICE_UNAVAILABLE,
                "queue_timeout",
                format!("the {class} request was not admitted within queue_timeout_s"),
            ),
            Refusal::Draining => self.retry(
                StatusCode::SERVICE_UNAVAILABLE,
                "draining",
                "the gateway handed its sockets to a successor before admitting the request; send it again",
            ),
        }
    }

    /// The 429 for a request the admission policy shed before its response began: its place went to a
    /// stricter class.
    pub fn shed(mut self) -> Response {
        self.outcome = Some(Outcome::Shed);
        self.error(
            StatusCode::TOO_MANY_REQUESTS,
            "shed",
            "shed by the admission policy to make room for a stricter class",
        )
    }

    /// The 503 for a background request that had no response yet at a drain's cut (`grace` into the drain):
    /// `Retry-After: 1`, for the gateway that serves the sockets next.
    pub fn drained(mut self, grace: Duration) -> Response {
        self.outcome = Some(Outcome::Drained);
        let message = crate::server::drain_message(grace);
        self.not_ready("draining", message, Duration::from_secs(1))
    }

    /// Records why a response already begun went wrong, for the request's log line.
    pub fn note_error(&mut self, message: String) {
        self.error = Some(message);
    }

    /// Records how a forwarded request ended.
    pub fn finish(mut self, outcome: Outcome) {
        self.outcome = Some(outcome);
    }

    /// The request's terminal outcome: a response relayed to its end is `completed` only with a 2xx status, and a
    /// request without one ended as its client left.
    fn terminal(&self) -> Outcome {
        match (self.outcome, self.status) {
            (Some(Outcome::Completed), Some(status)) if status.is_client_error() => {
                Outcome::UpstreamClientError
            }
            (Some(Outcome::Completed), Some(status)) if !status.is_success() => {
                Outcome::UpstreamServerError
            }
            (Some(outcome), _) => outcome,
            (None, None) => Outcome::Cancelled,
            (None, Some(status)) if status.is_client_error() => Outcome::Invalid,
            (None, Some(status)) if status.is_success() => Outcome::Completed,
            (None, Some(_)) => Outcome::TransportError,
        }
    }

    /// How the SLI counts a request of a class that ended with `outcome` after `total`: `None` when it is not valid,
    /// else whether it is good.
    fn judged(&self, outcome: Outcome, total: Duration) -> Option<bool> {
        let within = |first: Duration| self.objective.is_none_or(|objective| first <= objective);
        match outcome {
            outcome if outcome.clients_fault() => None,
            Outcome::Completed => Some(self.first_byte.is_some_and(within)),
            // A client that leaves is no failure of the service, unless the service had already met or missed the
            // objective: the request is judged by its first content, or bad if the objective passed without one.
            Outcome::Cancelled => match (self.objective, self.first_byte) {
                (Some(_), Some(first)) => Some(within(first)),
                (Some(objective), None) if total > objective => Some(false),
                _ => None,
            },
            _ => Some(false),
        }
    }
}

impl Drop for Exchange {
    fn drop(&mut self) {
        let status = self.status.map_or(CLIENT_CLOSED, |s| s.as_u16());
        let total = self.received.elapsed();
        let inference = self.route.is_inference();
        // Every inference request has an outcome, and a proxied model list one if it went upstream.
        let outcome = (inference || self.forwarded).then(|| self.terminal());
        self.offer();
        let metrics = &self.metrics;
        metrics.request(self.tenant.as_ref(), self.route, self.class, status);
        if let (true, Some(outcome)) = (inference, outcome) {
            metrics.terminal(self.tenant.as_ref(), self.route, self.class, outcome);
            if let Some(class) = self.class {
                metrics.outcome(class, outcome);
                if let Some(good) = self.judged(outcome, total) {
                    metrics.sli(class, good);
                }
            }
            if outcome == Outcome::Completed {
                self.succeeded();
            }
        }
        if let (Some(class), true) = (self.class, self.forwarded) {
            metrics.duration(class, total);
            if let Some(first_byte) = self.first_byte {
                metrics.first_byte(class, first_byte);
            }
            if let Some(usage) = &self.summary.usage {
                metrics.usage(class, usage);
            }
        }

        let usage = self.summary.usage;
        let ms = |d: Duration| d.as_micros() as f64 / 1e3;
        macro_rules! request_line {
            ($level:expr) => {
                tracing::event!(
                    target: "flatshare_gateway::request",
                    $level,
                    request_id = self.id.to_str().unwrap_or_default(),
                    tenant = self.tenant.as_deref(),
                    session = self.session.as_ref().map(SessionId::as_str),
                    node = self.node.as_deref(),
                    class = self.class.map(SloClass::as_str),
                    route = self.route.as_str(),
                    status,
                    outcome = outcome.map(Outcome::as_str),
                    admission_wait_ms = self.admission_wait.map(ms),
                    deferred = self.deferred.map(Deferral::as_str),
                    aged = self.aged.then_some(true),
                    degraded = self.degraded.then_some(true),
                    resubmits = (self.resubmits > 0).then_some(self.resubmits),
                    retries = (self.retries > 0).then_some(self.retries),
                    first_byte_ms = self.first_byte.map(ms),
                    total_ms = ms(total),
                    prompt_tokens = usage.map(|u| u.prompt),
                    completion_tokens = usage.map(|u| u.completion),
                    cached_tokens = usage.map(|u| u.cached),
                    finish_reason = self.summary.finish_reason.as_deref(),
                    error = self.error.as_deref(),
                    "request"
                )
            };
        }
        let failed = matches!(
            outcome,
            Some(
                Outcome::TransportError
                    | Outcome::Stalled
                    | Outcome::TimeoutHeaders
                    | Outcome::TimeoutFirstContent
                    | Outcome::TimeoutIdle
                    | Outcome::TimeoutTotal
            )
        );
        if status >= 500 || failed {
            request_line!(Level::WARN);
        } else {
            request_line!(Level::INFO);
        }
    }
}
