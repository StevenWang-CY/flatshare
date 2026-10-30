//! The request path's vocabulary, each as the label its metrics carry: the API routes, how an inference request
//! ended, why the gateway ended an event stream, why an upstream request failed, and the engine's state.

use crate::deadline::Timeout;

/// The API routes, as the `route` label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    ChatCompletions,
    Completions,
    Responses,
    /// `POST /v1/messages`, the Anthropic Messages API, as Claude Code sends it.
    Messages,
    /// `POST /v1/messages/count_tokens`: a Messages body's prompt tokens, generating nothing.
    CountTokens,
    Models,
    Events,
    /// `POST /v1/sessions`, with `[[nodes]]`.
    Sessions,
    /// `DELETE /v1/sessions/{id}`, with `[[nodes]]`.
    Session,
    /// Any path or method the gateway does not serve.
    Other,
}

impl Route {
    /// The routes that proxy a request body, whose overhead and upload are timed.
    pub(crate) const INFERENCE: [Self; 5] = [
        Self::ChatCompletions,
        Self::Completions,
        Self::Responses,
        Self::Messages,
        Self::CountTokens,
    ];

    /// Whether the route is an inference route, whose requests are offered, admitted and ended.
    pub fn is_inference(self) -> bool {
        Self::INFERENCE.contains(&self)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
            Self::Responses => "/v1/responses",
            Self::Messages => "/v1/messages",
            Self::CountTokens => "/v1/messages/count_tokens",
            Self::Models => "/v1/models",
            Self::Events => "/v1/events",
            Self::Sessions => "/v1/sessions",
            Self::Session => "/v1/sessions/{id}",
            Self::Other => "other",
        }
    }
}

/// How an inference request ended: its terminal event, one a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// A 2xx response relayed to its end.
    Completed,
    /// The engine answered 4xx (the request's own fault, such as a prompt over the context), relayed to its end.
    UpstreamClientError,
    /// The engine answered another status than 2xx or 4xx, relayed to its end.
    UpstreamServerError,
    /// The upstream failed: no connection, one that closed before the response headers, or a body that broke off.
    TransportError,
    /// The client went away first; the gateway closed the upstream request.
    Cancelled,
    /// The client stopped reading for longer than the stall timeout; the gateway aborted both sides.
    Stalled,
    /// The admission policy shed it to make room for a stricter class; the gateway cancelled the upstream
    /// request.
    Shed,
    /// A draining gateway cut it: a background request still running `background_drain_s` into a drain (a
    /// `draining` error event on a stream, or a 503 before the response), or one a gateway that handed its sockets
    /// over did not admit.
    Drained,
    /// One of the request's deadlines passed ([`crate::deadline`]); the gateway cancelled the upstream request.
    TimeoutUpload,
    TimeoutQueue,
    TimeoutHeaders,
    TimeoutFirstContent,
    TimeoutIdle,
    TimeoutTotal,
    /// The intake refused it before reading its body ([`crate::intake`]): the gateway, its class or its tenant had
    /// its requests in flight, or the bodies' budget was full.
    RefusedInFlight,
    RefusedClass,
    RefusedTenant,
    RefusedBodyBytes,
    /// The admission point's queues were full.
    RefusedQueue,
    /// The engine was not ready: a request had found it broken and its `/health` had not answered since. With
    /// `[[nodes]]`, also a request that no node could take (`no_healthy_node`, `no_call_slot`, `no_agent_slot`,
    /// `placement_full`), its 503 naming which.
    NotReady,
    /// The gateway refused it as malformed or unauthorized: 400, 401, 404, 405 or 413.
    Invalid,
}

impl Outcome {
    pub(crate) const ALL: [Self; 21] = [
        Self::Completed,
        Self::UpstreamClientError,
        Self::UpstreamServerError,
        Self::TransportError,
        Self::Cancelled,
        Self::Stalled,
        Self::Shed,
        Self::Drained,
        Self::TimeoutUpload,
        Self::TimeoutQueue,
        Self::TimeoutHeaders,
        Self::TimeoutFirstContent,
        Self::TimeoutIdle,
        Self::TimeoutTotal,
        Self::RefusedInFlight,
        Self::RefusedClass,
        Self::RefusedTenant,
        Self::RefusedBodyBytes,
        Self::RefusedQueue,
        Self::NotReady,
        Self::Invalid,
    ];

    /// The outcome of a request whose `timeout` passed.
    pub const fn timeout(timeout: Timeout) -> Self {
        match timeout {
            Timeout::Upload => Self::TimeoutUpload,
            Timeout::Queue => Self::TimeoutQueue,
            Timeout::Headers => Self::TimeoutHeaders,
            Timeout::FirstContent => Self::TimeoutFirstContent,
            Timeout::Idle => Self::TimeoutIdle,
            Timeout::Total => Self::TimeoutTotal,
        }
    }

    /// Whether the client is at fault, which the SLI leaves out of its valid requests: a malformed or unauthorized
    /// request, one the engine answered 4xx, an upload too slow, a tenant over its own bound, a client that stopped
    /// reading. A client's cancellation is judged by the request's own state ([`crate::exchange`]).
    pub const fn clients_fault(self) -> bool {
        matches!(
            self,
            Self::Invalid
                | Self::UpstreamClientError
                | Self::TimeoutUpload
                | Self::RefusedTenant
                | Self::Stalled
        )
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::UpstreamClientError => "upstream_client_error",
            Self::UpstreamServerError => "upstream_server_error",
            Self::TransportError => "transport_error",
            Self::Cancelled => "cancelled",
            Self::Stalled => "stalled",
            Self::Shed => "shed",
            Self::Drained => "drained",
            Self::TimeoutUpload => "timeout_upload",
            Self::TimeoutQueue => "timeout_queue",
            Self::TimeoutHeaders => "timeout_headers",
            Self::TimeoutFirstContent => "timeout_first_content",
            Self::TimeoutIdle => "timeout_idle",
            Self::TimeoutTotal => "timeout_total",
            Self::RefusedInFlight => "refused_in_flight",
            Self::RefusedClass => "refused_class",
            Self::RefusedTenant => "refused_tenant",
            Self::RefusedBodyBytes => "refused_body_bytes",
            Self::RefusedQueue => "refused_queue",
            Self::NotReady => "not_ready",
            Self::Invalid => "invalid",
        }
    }
}

/// The engine's state, as the gateway's inference canary and its requests find it ([`crate::canary`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EngineState {
    /// No inference has succeeded yet.
    Warming,
    /// Inference succeeds.
    Serving,
    /// The canary failed `degraded_after` times in a row with nothing succeeding.
    Degraded,
    /// The gateway is draining.
    Drained,
}

impl EngineState {
    pub const ALL: [Self; 4] = [Self::Warming, Self::Serving, Self::Degraded, Self::Drained];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Warming => "warming",
            Self::Serving => "serving",
            Self::Degraded => "degraded",
            Self::Drained => "drained",
        }
    }
}

/// Why the gateway ended an event stream with an error event: its `type`, and the HTTP status as its `code`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorEvent {
    /// The upstream broke the stream off (502).
    Upstream,
    /// A draining gateway cut a background stream (503): the client may send it again.
    Draining,
    /// A deadline passed while the upstream streamed (504): it stopped sending, or the request ran too long.
    Timeout,
}

impl ErrorEvent {
    pub(crate) const ALL: [Self; 3] = [Self::Upstream, Self::Draining, Self::Timeout];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Upstream => "upstream_unavailable",
            Self::Draining => "draining",
            Self::Timeout => "upstream_timeout",
        }
    }

    pub const fn code(self) -> u16 {
        match self {
            Self::Upstream => 502,
            Self::Draining => 503,
            Self::Timeout => 504,
        }
    }
}

/// Why an upstream request failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UpstreamError {
    /// No connection could be made (502).
    Connect,
    /// The connection failed before the response headers arrived (502).
    Request,
    /// A deadline passed while the gateway waited on the upstream: its headers, its first content, a frame after
    /// one, or the request's total (504, or an error event mid-stream).
    Timeout,
    /// The response body broke off.
    Body,
}

impl UpstreamError {
    pub(crate) const ALL: [Self; 4] = [Self::Connect, Self::Request, Self::Timeout, Self::Body];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::Request => "request",
            Self::Timeout => "timeout",
            Self::Body => "body",
        }
    }
}
