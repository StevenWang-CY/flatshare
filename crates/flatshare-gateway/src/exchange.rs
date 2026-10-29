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
use crate::config::ClassConfig;
use crate::deadline::{Deadlines, Phase, Timeout};
use crate::harness::{NotAdmitted, Refusal, Ticket};
use crate::intake::{Permit, Refused};
use crate::metrics::Metrics;
use crate::nodes::{Held, Placed, Unplaced};
use crate::sanitize::{InvalidBody, REQUEST_ID, RESOLVED_SESSION_ID, request_id, session_header};
use crate::types::{Outcome, Route, UpstreamError};
use crate::upstream::RECOVERY_PROBE;
use crate::usage::Summary;

/// The status recorded for a request whose client left before any response: nginx's "client closed request".
const CLIENT_CLOSED: u16 = 499;

/// The message of the 503 that a request gets while the engine is not ready.
const NOT_READY: &str =
    "the engine is not ready: a request found it broken and its /health has not answered since";

/// An error answer's status, code and message, as [`Exchange::error`] takes them: what a step that cannot answer
/// gives the record to answer.
pub(crate) type ErrorAnswer = (StatusCode, &'static str, String);

/// The 401 of a request without a known API key on a listener without a tenant.
pub(crate) fn missing_key() -> ErrorAnswer {
    let message = "missing or unknown API key".to_owned();
    (StatusCode::UNAUTHORIZED, "invalid_api_key", message)
}

#[derive(Debug)]
pub(crate) struct Exchange {
    metrics: Arc<Metrics>,
    /// Sent upstream as `X-Request-Id` and returned to the client.
    pub id: HeaderValue,
    pub route: Route,
    pub received: Instant,
    /// What the gateway resolved of the request.
    pub resolved: Resolved,
    /// The status sent to the client; `None` until a response exists.
    pub status: Option<StatusCode>,
    /// How long the request waited before it went upstream: at the admission point, or held back by a node that
    /// covers. `None` when it waited at neither.
    pub admission_wait: Option<Duration>,
    pub first_byte: Option<Duration>,
    pub summary: Summary,
    /// The request's place in the intake and its body's bytes, given back when the record ends.
    pub permit: Option<Permit>,
    /// Where the request went.
    pub routed: Routed,
    /// What E2's harness did with the request, under an admission policy.
    pub harness: Harness,
    /// Whether the request was sent upstream, which makes it count towards outcomes.
    forwarded: bool,
    /// Whether it was counted as offered.
    offered: bool,
    /// How it ended, once a step knows; set only by [`Exchange::ends_as`] and [`Exchange::ends_as_unless_set`].
    outcome: Option<Outcome>,
    error: Option<String>,
}

/// What the gateway resolved of a request: its tenant, class and session, and its class's deadlines and objective.
#[derive(Debug, Default)]
pub(crate) struct Resolved {
    pub tenant: Option<Arc<str>>,
    pub class: Option<SloClass>,
    pub session: Option<SessionId>,
    /// Its deadlines, once its class is known.
    pub deadlines: Option<Deadlines>,
    /// Its class's first-content objective, for the SLI.
    pub objective: Option<Duration>,
}

/// Where a request went: its engine, and with `[[nodes]]` its node.
#[derive(Debug, Default)]
pub(crate) struct Routed {
    /// The engine's state, which an inference request's success keeps serving.
    pub health: Option<Arc<Health>>,
    /// With `[[nodes]]`: the request's node, for its log line, and its place there, given back when the record
    /// ends.
    pub node: Option<Arc<str>>,
    pub placed: Option<Placed>,
    /// How many times it was sent again on another node after its node failed before its first content.
    pub retries: u32,
}

/// What E2's harness did with a request: the lifecycle policy's deferral, aging and degrade, and preempt-resubmit's
/// resubmissions.
#[derive(Debug, Default)]
pub(crate) struct Harness {
    /// Why the lifecycle policy deferred it.
    pub deferred: Option<Deferral>,
    /// Whether the lifecycle policy admitted it regardless, after `max_defer_s`.
    pub aged: bool,
    /// Whether the lifecycle policy's degrade capped its reply.
    pub degraded: bool,
    /// How many times it was sent upstream again after a preemption.
    pub resubmits: u32,
}

impl Exchange {
    pub fn new(metrics: &Arc<Metrics>, route: Route) -> Self {
        Self {
            metrics: Arc::clone(metrics),
            id: request_id(),
            route,
            received: Instant::now(),
            resolved: Resolved::default(),
            status: None,
            admission_wait: None,
            first_byte: None,
            summary: Summary::default(),
            permit: None,
            routed: Routed::default(),
            harness: Harness::default(),
            forwarded: false,
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
            let Resolved { tenant, class, .. } = &self.resolved;
            self.metrics.offered(tenant.as_ref(), self.route, *class);
        }
    }

    /// Records the class of an inference request as it is known: the request is offered, with the class's deadlines
    /// from its receipt and its first-content objective. Returns the deadlines.
    pub fn classify(&mut self, class: SloClass, config: &ClassConfig) -> Deadlines {
        self.resolved.class = Some(class);
        self.offer();
        let deadlines = Deadlines::new(class, config, self.received);
        self.resolved.deadlines = Some(deadlines);
        self.resolved.objective = config.first_content_objective;
        deadlines
    }

    /// The request goes upstream, which makes it count towards outcomes. An inference request is counted as
    /// admitted: past the intake, the admission point and the readiness check; `policy` when an admission policy
    /// decided when.
    pub fn admit(&mut self, policy: bool) {
        self.forwarded = true;
        if self.route.is_inference() {
            let Resolved { tenant, class, .. } = &self.resolved;
            self.metrics
                .admitted(tenant.as_ref(), self.route, *class, policy);
        }
    }

    /// Records the request's place on a node: the node, for its log line, and its engine's state, which the request's
    /// success keeps serving. A place it had before goes.
    pub fn place_on(&mut self, placed: Placed) {
        self.routed.node = Some(Arc::clone(&placed.node.name));
        self.routed.health = Some(Arc::clone(&placed.node.engine.health));
        self.routed.placed = Some(placed);
    }

    /// An inference succeeded: a 2xx response's first content arrived, or it was relayed to its end.
    pub fn succeeded(&self) {
        if let Some(health) = &self.routed.health {
            health.succeeded();
        }
    }

    /// Adds the request id and, once resolved, the session id to response headers.
    pub fn decorate(&self, headers: &mut HeaderMap) {
        headers.insert(REQUEST_ID.clone(), self.id.clone());
        if let Some(session) = &self.resolved.session {
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

    /// Answers with an OpenAI-format error, `{"error": {"message", "type", "code"}}`, and records it. Its outcome,
    /// unless the caller set one first: `transport_error` once the request went upstream, `invalid` for a client
    /// error before.
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
            self.ends_as_unless_set(Outcome::TransportError);
        } else if status.is_client_error() {
            self.ends_as_unless_set(Outcome::Invalid);
        }
        self.error = Some(message);
        self.respond(status, &body)
    }

    /// The 503 for a request that no ready engine can take, sent without going upstream. `Retry-After` is `retry`,
    /// the interval of the readiness probe that will say when an engine answers again, in whole seconds. Its outcome
    /// is `not_ready`, unless the caller set one first (a drain's is `drained`).
    pub fn not_ready(
        mut self,
        code: &str,
        message: impl Into<String>,
        retry: Duration,
    ) -> Response {
        self.ends_as_unless_set(Outcome::NotReady);
        let mut response = self.error(StatusCode::SERVICE_UNAVAILABLE, code, message);
        let seconds = retry.as_secs_f64().ceil().max(1.0) as u64;
        response
            .headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from(seconds));
        response
    }

    /// The 503 for a request while the gateway's one engine is not ready, with `Retry-After` the interval of the
    /// probe that will say when it answers again ([`crate::upstream::recover`]).
    pub fn engine_not_ready(self) -> Response {
        self.not_ready("upstream_not_ready", NOT_READY, RECOVERY_PROBE)
    }

    /// The 400 for a body whose reply limits or choices the gateway does not forward ([`crate::bounds`]), with the
    /// field at fault as the error's `param`. Its outcome is `invalid`, over any other.
    pub fn invalid_field(mut self, invalid: &bounds::Invalid) -> Response {
        let message = invalid.to_string();
        let body = serde_json::json!({"error": {
            "message": message, "type": "invalid_request_error", "param": invalid.field(), "code": invalid.code(),
        }});
        self.ends_as(Outcome::Invalid);
        self.error = Some(message);
        self.respond(StatusCode::BAD_REQUEST, &body)
    }

    /// The 400 for a body the gateway does not rewrite: not a JSON object, or with a reply limit or choices it does
    /// not forward ([`crate::bounds`]).
    pub fn invalid_body(self, invalid: &InvalidBody) -> Response {
        match invalid {
            InvalidBody::Json(_) => {
                self.error(StatusCode::BAD_REQUEST, "invalid_json", invalid.to_string())
            }
            InvalidBody::Bounds(bounds) => self.invalid_field(bounds),
        }
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
    /// when the gateway admits nothing more; each with `Retry-After: 1`. Its outcome, over any other, is the
    /// refusal's.
    pub fn intake_refused(mut self, refused: Refused, message: String) -> Response {
        let (status, code) = match refused {
            Refused::Draining => (StatusCode::SERVICE_UNAVAILABLE, "draining"),
            Refused::InFlight => (StatusCode::SERVICE_UNAVAILABLE, "in_flight_limit"),
            Refused::Class => (StatusCode::SERVICE_UNAVAILABLE, "class_in_flight_limit"),
            Refused::Tenant => (StatusCode::TOO_MANY_REQUESTS, "tenant_in_flight_limit"),
            Refused::BodyBytes => (StatusCode::SERVICE_UNAVAILABLE, "body_budget_exhausted"),
        };
        self.ends_as(match refused {
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
    /// only an attempt that was dropped), 504 for the upstream. Its outcome, over any other, is the deadline's.
    pub fn timed_out(mut self, phase: Phase, timeout: Timeout) -> Response {
        let message = self.resolved.deadlines.map_or_else(
            || format!("the request passed its {}", timeout.key()),
            |deadlines| deadlines.message(timeout),
        );
        self.ends_as(Outcome::timeout(timeout));
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
    /// table whose every session is busy. Its outcome is `not_ready`.
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
        if let Some(placed) = &self.routed.placed {
            placed.node.engine.found_broken();
        }
    }

    /// The 401 for a request without a known API key on a listener without a tenant ([`missing_key`]).
    pub fn unauthorized(self) -> Response {
        let (status, code, message) = missing_key();
        self.error(status, code, message)
    }

    /// The error for a request the admission point refused: 429 when its class's queue was full, 503 when it
    /// was not admitted in time. Its outcome, over any other: `refused_queue`, `timeout_queue`, or `drained` for a
    /// gateway that handed its sockets over.
    pub fn refused(mut self, refusal: Refusal) -> Response {
        let class = self.resolved.class.map_or("", SloClass::as_str);
        self.ends_as(match refusal {
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

    /// Records the wait of a request the admission point admitted, with its `ticket` (`None` for a class the policy
    /// does not hold, admitted at once), and why the lifecycle policy deferred it or admitted it regardless. Returns
    /// how long it waited.
    pub fn admitted_after(&mut self, ticket: Option<&Ticket>) -> Duration {
        let waited = ticket.map_or(Duration::ZERO, Ticket::waited);
        self.admission_wait = Some(waited);
        if let Some(ticket) = ticket {
            self.harness.deferred = ticket.deferred();
            self.harness.aged = ticket.aged();
        }
        waited
    }

    /// The answer to a request that went no further after `waited` at the admission point: refused, past its
    /// deadline, or cut by a drain `grace` into it.
    pub fn not_admitted(mut self, not: NotAdmitted, waited: Duration, grace: Duration) -> Response {
        self.admission_wait = Some(waited);
        match not {
            NotAdmitted::Refused(refusal) => self.refused(refusal),
            NotAdmitted::Late(timeout) => self.timed_out(Phase::Queue, timeout),
            NotAdmitted::Cut => self.drained(grace),
        }
    }

    /// Records how long a covering node held the request back, if at all. Returns it.
    pub fn held_back(&mut self, waited: Duration) -> Duration {
        if !waited.is_zero() {
            self.admission_wait = Some(waited);
        }
        waited
    }

    /// The answer to a request that a covering node held back for `waited`, until its deadline or a drain's cut
    /// `grace` into the drain.
    pub fn not_released(mut self, held: Held, waited: Duration, grace: Duration) -> Response {
        self.admission_wait = Some(waited);
        match held {
            Held::Late(timeout) => self.timed_out(Phase::Queue, timeout),
            Held::Cut => self.drained(grace),
        }
    }

    /// The 429 for a request the admission policy shed before its response began: its place went to a
    /// stricter class. Its outcome is `shed`, over any other.
    pub fn shed(mut self) -> Response {
        self.ends_as(Outcome::Shed);
        self.error(
            StatusCode::TOO_MANY_REQUESTS,
            "shed",
            "shed by the admission policy to make room for a stricter class",
        )
    }

    /// The 503 for a background request that had no response yet at a drain's cut (`grace` into the drain):
    /// `Retry-After: 1`, for the gateway that serves the sockets next. Its outcome is `drained`, over any other.
    pub fn drained(mut self, grace: Duration) -> Response {
        self.ends_as(Outcome::Drained);
        let message = crate::drain::drain_message(grace);
        self.not_ready("draining", message, Duration::from_secs(1))
    }

    /// Records why a response already begun went wrong, for the request's log line.
    pub fn note_error(&mut self, message: String) {
        self.error = Some(message);
    }

    /// Records how a forwarded request ended: `outcome`, over any other.
    pub fn finish(mut self, outcome: Outcome) {
        self.ends_as(outcome);
    }

    /// Sets how the request ended, over what an earlier step set: the caller knows.
    fn ends_as(&mut self, outcome: Outcome) {
        self.outcome = Some(outcome);
    }

    /// Sets how the request ended unless an earlier step set it: the caller knows only its answer, and a step that
    /// knew more set its outcome first.
    fn ends_as_unless_set(&mut self, outcome: Outcome) {
        self.outcome.get_or_insert(outcome);
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
        let objective = self.resolved.objective;
        let within = |first: Duration| objective.is_none_or(|objective| first <= objective);
        match outcome {
            outcome if outcome.clients_fault() => None,
            Outcome::Completed => Some(self.first_byte.is_some_and(within)),
            // A client that leaves is no failure of the service, unless the service had already met or missed the
            // objective: the request is judged by its first content, or bad if the objective passed without one.
            Outcome::Cancelled => match (objective, self.first_byte) {
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
        let Resolved {
            tenant,
            class,
            session,
            ..
        } = &self.resolved;
        metrics.request(tenant.as_ref(), self.route, *class, status);
        if let (true, Some(outcome)) = (inference, outcome) {
            metrics.terminal(tenant.as_ref(), self.route, *class, outcome);
            if let Some(class) = *class {
                metrics.outcome(class, outcome);
                if let Some(good) = self.judged(outcome, total) {
                    metrics.sli(class, good);
                }
            }
            if outcome == Outcome::Completed {
                self.succeeded();
            }
        }
        if let (Some(class), true) = (*class, self.forwarded) {
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
                    tenant = tenant.as_deref(),
                    session = session.as_ref().map(SessionId::as_str),
                    node = self.routed.node.as_deref(),
                    class = class.map(SloClass::as_str),
                    route = self.route.as_str(),
                    status,
                    outcome = outcome.map(Outcome::as_str),
                    admission_wait_ms = self.admission_wait.map(ms),
                    deferred = self.harness.deferred.map(Deferral::as_str),
                    aged = self.harness.aged.then_some(true),
                    degraded = self.harness.degraded.then_some(true),
                    resubmits = (self.harness.resubmits > 0).then_some(self.harness.resubmits),
                    retries = (self.routed.retries > 0).then_some(self.routed.retries),
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

#[cfg(test)]
mod tests {
    use http_body_util::BodyExt;

    use super::*;
    use crate::types::ErrorEvent;

    /// `tests/fixtures/errors.json`: the errors the voice worker reads, as the gateway sends them. An answer's
    /// `code` says what failed, while its `type` is the OpenAI class of its status. A stream's error event says it in
    /// its `type`, and its `code` is the status. `python/flatshare_voice`'s tests send these bodies from their fake
    /// gateways. `FLATSHARE_GATEWAY_WRITE_FIXTURES=1` writes it anew (`make gateway-fixtures`).
    #[tokio::test]
    async fn the_errors_fixture_is_current() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/errors.json");
        let metrics = Arc::new(Metrics::new());
        let placement = || Exchange::new(&metrics, Route::Sessions);
        let chat = || Exchange::new(&metrics, Route::ChatCompletions);
        let mut answers = Vec::new();
        for response in [
            // A gateway without [[nodes]]: the fallback's answer to POST /v1/sessions.
            Exchange::new(&metrics, Route::Other).error(
                StatusCode::NOT_FOUND,
                "not_found",
                "no such route: POST /v1/sessions",
            ),
            placement().unplaced(Unplaced::NoCallSlot, SloClass::Realtime),
            placement().unplaced(Unplaced::NoNode, SloClass::Realtime),
            placement().unplaced(Unplaced::Full, SloClass::Realtime),
            chat().error(
                StatusCode::BAD_GATEWAY,
                "upstream_unavailable",
                "the upstream response broke off before its first byte",
            ),
            chat().timed_out(Phase::Upstream, Timeout::FirstContent),
            chat().unplaced(Unplaced::NoNode, SloClass::Realtime),
            chat().intake_refused(
                Refused::InFlight,
                "the gateway has max_in_flight requests in flight".to_owned(),
            ),
            chat().error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                "the request body is not valid JSON",
            ),
        ] {
            let status = response.status().as_u16();
            let body = response.into_body().collect().await.unwrap().to_bytes();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            let entry =
                serde_json::json!({"status": status, "code": body["error"]["code"], "body": body});
            answers.push(serde_json::to_string(&entry).unwrap());
        }
        let events: Vec<String> = [
            ErrorEvent::Upstream,
            ErrorEvent::Timeout,
            ErrorEvent::Draining,
        ]
        .into_iter()
        .map(|kind| {
            let data = crate::relay::error_event(kind, "the upstream stream broke off", true);
            let entry = serde_json::json!({
                "type": kind.as_str(), "data": String::from_utf8(data.to_vec()).unwrap(),
            });
            serde_json::to_string(&entry).unwrap()
        })
        .collect();
        let about = "The gateway's errors that the voice worker reads (src/exchange.rs, src/relay.rs). Each answer has \
                     its status and its body: the error's code says what failed, and its type is the OpenAI class \
                     of its status. Each event is the line a stream ends with, whose type says what failed and whose \
                     code is the status. Written by `make gateway-fixtures`, checked by the gateway's tests, and sent \
                     by the fake gateways of python/flatshare_voice's tests. The messages are examples.";
        let fixture = format!(
            "{{\n  \"about\": {},\n  \"answers\": [\n    {}\n  ],\n  \"events\": [\n    {}\n  ]\n}}\n",
            serde_json::to_string(about).unwrap(),
            answers.join(",\n    "),
            events.join(",\n    ")
        );
        if std::env::var_os("FLATSHARE_GATEWAY_WRITE_FIXTURES").is_some() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &fixture).unwrap();
        }
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            on_disk == fixture,
            "{} is stale; write it anew with `make gateway-fixtures`",
            path.display()
        );
        // The two shapes differ in where the failure is named, which is what the worker must read.
        let parsed: serde_json::Value = serde_json::from_str(&fixture).unwrap();
        let first = &parsed["answers"][1]["body"]["error"];
        assert_eq!(
            (first["type"].as_str(), first["code"].as_str()),
            (Some("server_error"), Some("no_call_slot"))
        );
        assert!(
            parsed["events"][0]["data"]
                .as_str()
                .unwrap()
                .contains(r#""type":"upstream_unavailable""#)
        );
    }
}
