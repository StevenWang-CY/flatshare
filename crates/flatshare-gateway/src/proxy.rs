//! The proxy path: authenticate, sanitize, resolve, admit, and send the request upstream. The engine and its
//! readiness are [`crate::upstream`]'s, and the response's relay to the client [`crate::relay`]'s.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::Request;
use axum::response::Response;
use bytes::Bytes;
use bytes::BytesMut;
use flatshare_proto::SloClass;
use http::header::{CONTENT_TYPE, HeaderValue};
use http::uri::PathAndQuery;
use http::{StatusCode, Version};
use http_body::Body as _;
use http_body_util::BodyExt;
use hyper::body::Incoming;

use crate::deadline::{Deadlines, Phase, Timeout};
use crate::exchange::Exchange;
use crate::harness::{Refusal, Ticket};
use crate::intake::{BodyBytes, Refused};
use crate::metrics::{Route, UpstreamError};
use crate::relay;
use crate::sanitize::{
    Bounds, InvalidBody, REQUEST_ID, cap_reply, resolve_class, resolve_session, rewrite_body,
    sanitize_headers, set_engine_headers, strip_hop_by_hop,
};
use crate::server::{self, App, Shared};
use crate::upstream::{RECOVERY_PROBE, Upstream, UpstreamBody, causes, recover};

/// The message of the 503 that a request gets while the engine is not ready.
const NOT_READY: &str =
    "the engine is not ready: a request found it broken and its /health has not answered since";

/// `POST /v1/chat/completions`, `/v1/completions` and `/v1/responses`.
pub(crate) async fn inference(app: App, route: Route, request: Request) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, route);
    let (mut parts, body) = request.into_parts();
    let Some(tenant) = shared
        .tenants
        .authenticate(&parts.headers, app.listener_tenant)
    else {
        return ex.unauthorized();
    };
    ex.tenant = Some(Arc::clone(&tenant.name));

    let inputs = sanitize_headers(&mut parts.headers, tenant.trusted, &shared.metrics);
    let class = match resolve_class(tenant, inputs.class.as_ref()) {
        Ok((class, clamped)) => {
            if clamped {
                shared.metrics.class_clamped();
            }
            class
        }
        Err(err) => return ex.error(StatusCode::BAD_REQUEST, "invalid_class", err.to_string()),
    };
    ex.class = Some(class);
    let deadlines = Deadlines::new(class, shared.class(class), ex.received);
    ex.deadlines = Some(deadlines);

    // The intake's bounds, before a byte of the body is read: a refused request costs the gateway no memory and no
    // wait.
    let mut permit = match shared.intake.admit(tenant, class) {
        Ok(permit) => permit,
        Err(refused) => {
            let message = shared.intake.message(refused, tenant, class);
            return ex.intake_refused(refused, message);
        }
    };
    let upload = deadlines.upload();
    let body = match read_body(body, shared.max_body_bytes, &mut permit.body, upload).await {
        Ok(body) => body,
        Err(BodyError::Refused(refused)) => {
            let message = shared.intake.message(refused, tenant, class);
            return ex.intake_refused(refused, message);
        }
        Err(BodyError::Late(timeout)) => return ex.timed_out(Phase::Upload, timeout),
        Err(BodyError::Invalid(status, code, message)) => return ex.error(status, code, message),
    };
    // The request holds its place, and its body's bytes, until its record ends.
    ex.permit = Some(permit);
    // The overhead starts once the body is in: until then the gateway waits on the client, as the engine
    // would.
    let read = Instant::now();
    shared.metrics.request_body(route, read - ex.received);
    let signals = shared.upstream_priority;
    let priority = signals.body.then(|| class.engine_priority());
    // The class's reply cap and the choices, in the rewrite's one pass over the body.
    let bounds = Bounds {
        route,
        cap: shared.class(class).max_reply_tokens,
        max_choices: shared.max_choices,
    };
    let mut body = match rewrite_body(&body, &tenant.cache_salt, priority, bounds, &shared.metrics)
    {
        Ok(body) => body,
        Err(InvalidBody::Json(err)) => {
            let message = InvalidBody::Json(err).to_string();
            return ex.error(StatusCode::BAD_REQUEST, "invalid_json", message);
        }
        Err(InvalidBody::Bounds(invalid)) => return ex.invalid_field(&invalid),
    };
    let session = resolve_session(tenant, &inputs, body.prompt_cache_key.as_deref());
    set_engine_headers(&mut parts.headers, &session, class, &ex.id, signals);
    parts
        .headers
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let session = &*ex.session.insert(session);
    if !shared.upstream.ready() {
        return ex.not_ready("upstream_not_ready", NOT_READY, RECOVERY_PROBE);
    }

    // The admission point. Its decisions are the gateway's overhead; the time a request waits there is not.
    let mut start = read;
    let ticket = match &shared.admission {
        None => None,
        Some(admission) => {
            let arrived = Instant::now();
            // A background request still waiting at the drain's cut leaves the queue, and so does one whose wait
            // reaches its deadline.
            let mut cut = shared.drain.cut_for(Some(class));
            let (deadline, timeout) = deadlines.queue(arrived);
            let admitted = tokio::select! {
                biased;
                () = server::cut(&mut cut) => Waited::Cut,
                () = tokio::time::sleep_until(deadline.into()) => Waited::Late,
                admitted = admission.admit(tenant, class, session, body.json.len()) => Waited::Done(admitted),
            };
            let admitted = match admitted {
                Waited::Done(admitted) => admitted,
                Waited::Cut => {
                    ex.admission_wait = Some(arrived.elapsed());
                    return ex.drained(shared.drain.grace);
                }
                Waited::Late => {
                    ex.admission_wait = Some(arrived.elapsed());
                    return ex.timed_out(Phase::Queue, timeout);
                }
            };
            match admitted {
                Ok(ticket) => {
                    let waited = ticket.as_ref().map_or(Duration::ZERO, Ticket::waited);
                    ex.admission_wait = Some(waited);
                    if let Some(ticket) = &ticket {
                        ex.deferred = ticket.deferred();
                        ex.aged = ticket.aged();
                    }
                    start += waited;
                    ticket
                }
                Err(refusal) => {
                    ex.admission_wait = Some(arrived.elapsed());
                    return ex.refused(refusal);
                }
            }
        }
    };

    // Degrade's reply cap, over the class's: the lower of the two holds.
    let degrade = ticket.as_ref().and_then(Ticket::reply_cap);
    if let Some(cap) = degrade {
        body.json = cap_reply(&body.json, route, cap);
    }
    ex.degraded = degrade.is_some();

    let body = UpstreamBody::new(body.json, start, shared.metrics.overhead(route));
    forward(&app, ex, parts, body, ticket).await
}

/// `GET /v1/models`: authenticated and sanitized, with no session or class.
pub(crate) async fn models(app: App, request: Request) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, Route::Models);
    let (mut parts, _) = request.into_parts();
    let Some(tenant) = shared
        .tenants
        .authenticate(&parts.headers, app.listener_tenant)
    else {
        return ex.unauthorized();
    };
    ex.tenant = Some(Arc::clone(&tenant.name));
    // A model list has no class, and the interactive class's deadlines.
    let interactive = SloClass::Interactive;
    ex.deadlines = Some(Deadlines::new(
        interactive,
        shared.class(interactive),
        ex.received,
    ));
    sanitize_headers(&mut parts.headers, tenant.trusted, &shared.metrics);
    parts.headers.insert(REQUEST_ID.clone(), ex.id.clone());
    if !shared.upstream.ready() {
        return ex.not_ready("upstream_not_ready", NOT_READY, RECOVERY_PROBE);
    }
    forward(&app, ex, parts, UpstreamBody::empty(), None).await
}

/// What a wait at the admission point ended with.
enum Waited<T> {
    Done(T),
    /// The drain's cut.
    Cut,
    /// The request's deadline.
    Late,
}

/// Why a request body was not read.
#[derive(Debug)]
pub(crate) enum BodyError {
    /// The client's: too large (413) or unreadable (400).
    Invalid(StatusCode, &'static str, String),
    /// The gateway's budget for the bodies it holds cannot take it (503).
    Refused(Refused),
    /// It had not arrived by its deadline (408).
    Late(Timeout),
}

/// Reads a request body of at most `limit` bytes by `deadline`, whose bytes `budget` takes from the gateway's budget
/// as they arrive: a declared length all at once, before anything is read, and the rest chunk by chunk.
pub(crate) async fn read_body(
    mut body: Body,
    limit: usize,
    budget: &mut BodyBytes,
    (deadline, timeout): (Instant, Timeout),
) -> Result<Bytes, BodyError> {
    let too_large = || {
        BodyError::Invalid(
            StatusCode::PAYLOAD_TOO_LARGE,
            "body_too_large",
            format!("the request body exceeds {limit} bytes"),
        )
    };
    // A declared length over the limit is refused before reading anything.
    let declared = usize::try_from(body.size_hint().lower()).unwrap_or(usize::MAX);
    if declared > limit {
        return Err(too_large());
    }
    budget.take(declared).map_err(BodyError::Refused)?;
    let mut read = BytesMut::with_capacity(declared);
    loop {
        let Ok(frame) = tokio::time::timeout_at(deadline.into(), body.frame()).await else {
            return Err(BodyError::Late(timeout));
        };
        let Some(frame) = frame else {
            break;
        };
        let frame = frame.map_err(|err| {
            BodyError::Invalid(
                StatusCode::BAD_REQUEST,
                "unreadable_body",
                format!("cannot read the request body: {err}"),
            )
        })?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        let total = read.len() + data.len();
        if total > limit {
            return Err(too_large());
        }
        if let Some(more) = total.checked_sub(budget.taken()) {
            budget.take(more).map_err(BodyError::Refused)?;
        }
        read.extend_from_slice(&data);
    }
    Ok(read.freeze())
}

/// Sends the request upstream and relays the response. An admitted request's ticket goes with the relay, and is
/// given back when the response ends. A request shed before its response answers 429; one preempted before its
/// first byte is sent again.
async fn forward(
    app: &App,
    mut ex: Exchange,
    mut parts: http::request::Parts,
    body: UpstreamBody,
    mut ticket: Option<Ticket>,
) -> Response {
    let shared = &app.shared;
    let upstream = &shared.upstream;
    let path = parts.uri.path_and_query().cloned();
    parts.uri = upstream.uri(path.unwrap_or_else(|| PathAndQuery::from_static("/")));
    parts.version = Version::HTTP_11;
    parts.extensions.clear();

    ex.forwarded = true;
    let deadlines = ex.deadlines.expect("a forwarded request has its deadlines");
    // A background request without a response at the drain's cut is dropped, as is a shed one or one whose deadline
    // passes: dropping the request future closes its connection, which aborts the engine request.
    let mut cut = shared.drain.cut_for(ex.class);
    let sending = async {
        match &mut ticket {
            Some(ticket) if ticket.preemptible() => {
                resubmitting(upstream, &mut ex, parts, body, ticket, &deadlines).await
            }
            Some(ticket) if ticket.sheddable() => {
                let request = http::Request::from_parts(parts, body);
                tokio::select! {
                    sent = send(upstream, request, &deadlines) => sent.map(|response| (response, None)),
                    () = ticket.shed() => Err(Failure::Shed),
                }
            }
            _ => send(upstream, http::Request::from_parts(parts, body), &deadlines)
                .await
                .map(|response| (response, None)),
        }
    };
    let sent = tokio::select! {
        biased;
        () = server::cut(&mut cut) => Err(Failure::Drained),
        sent = sending => sent,
    };
    let (response, first) = match sent {
        Ok(sent) => sent,
        Err(failure) => {
            // An engine that a request found broken is probed until it answers; meanwhile its requests are
            // answered 503 ([`recover`]).
            if failure.broken() && upstream.start_recovery() {
                tokio::spawn(recover(Arc::downgrade(&app.shared)));
            }
            return failure.answer(ex, shared);
        }
    };

    let (mut parts, upstream_body) = response.into_parts();
    strip_hop_by_hop(&mut parts.headers);
    ex.decorate(&mut parts.headers);
    ex.status = Some(parts.status);
    // The first frame of a preemptible response, read before the client was answered, goes first.
    let body = relay::start(
        shared,
        &parts.headers,
        upstream_body,
        first,
        ex,
        ticket,
        cut,
    );
    Response::from_parts(parts, body)
}

/// Why a request has no response to relay.
#[derive(Debug)]
enum Failure {
    /// The connection failed, or closed before the response headers (502).
    Upstream(hyper_util::client::legacy::Error),
    /// A deadline passed ([`crate::deadline`]): in the upstream's response (504), or waiting at the admission
    /// point again after a preemption (503).
    Late(Phase, Timeout),
    /// A preemptible request's response broke off before its first body frame (502).
    Body,
    /// A preempted request was not admitted again in time (503).
    Refused(Refusal),
    /// The admission policy shed the request before its response, for a stricter class (429).
    Shed,
    /// A background request had no response yet at a drain's cut (503).
    Drained,
}

impl Failure {
    /// Whether the upstream broke: no connection, or one that ended before the response began. A timeout is a
    /// slow engine, which its probe would find serving, and a refusal the admission point's.
    fn broken(&self) -> bool {
        matches!(self, Self::Upstream(_) | Self::Body)
    }

    /// The client's answer, with the upstream error counted.
    fn answer(self, ex: Exchange, shared: &Shared) -> Response {
        match self {
            Self::Upstream(err) => {
                let kind = if err.is_connect() {
                    UpstreamError::Connect
                } else {
                    UpstreamError::Request
                };
                ex.metrics().upstream_error(kind);
                let message = format!("upstream request failed: {}", causes(&err));
                ex.error(StatusCode::BAD_GATEWAY, "upstream_unavailable", message)
            }
            Self::Late(phase, timeout) => ex.timed_out(phase, timeout),
            Self::Body => {
                ex.metrics().upstream_error(UpstreamError::Body);
                let message = "the upstream response broke off before its first byte";
                ex.error(StatusCode::BAD_GATEWAY, "upstream_unavailable", message)
            }
            Self::Refused(refusal) => ex.refused(refusal),
            Self::Shed => ex.shed(),
            Self::Drained => ex.drained(shared.drain.grace),
        }
    }
}

/// Sends a request upstream and waits for its response headers, by the earliest of its deadlines that applies.
async fn send(
    upstream: &Upstream,
    request: http::Request<UpstreamBody>,
    deadlines: &Deadlines,
) -> Result<http::Response<Incoming>, Failure> {
    let (deadline, timeout) = deadlines.headers(Instant::now());
    match tokio::time::timeout_at(deadline.into(), upstream.request(request)).await {
        Ok(response) => response.map_err(Failure::Upstream),
        Err(_) => Err(Failure::Late(Phase::Upstream, timeout)),
    }
}

/// Sends a preemptible request and holds its response back until the first body frame. A preemption before
/// that frame drops the attempt, which closes its connection and aborts the engine request, and the request
/// goes again once it is admitted again. The client has seen nothing of the dropped attempt, so it gets one
/// response whose first byte comes later. Each attempt has its own wait for its headers, and the request its deadlines
/// from its receipt whatever the attempt: a retry never buys it more time, and comes only before the client has seen
/// anything.
async fn resubmitting(
    upstream: &Upstream,
    ex: &mut Exchange,
    parts: http::request::Parts,
    body: UpstreamBody,
    ticket: &mut Ticket,
    deadlines: &Deadlines,
) -> Result<(http::Response<Incoming>, Option<Bytes>), Failure> {
    let again = (parts.clone(), body.again());
    let mut request = http::Request::from_parts(parts, body);
    loop {
        tokio::select! {
            attempt = first_frame(upstream, request, deadlines) => {
                let (response, first) = attempt?;
                // A frame that arrives as the request is preempted belongs to the dropped attempt.
                if first.is_none() || ticket.first_byte() {
                    if first.is_some() {
                        ex.first_byte = Some(ex.received.elapsed());
                    }
                    return Ok((response, first));
                }
            }
            () = ticket.preempted() => {}
        }
        // Waiting at the admission point again, within the request's own deadlines.
        let (deadline, timeout) = deadlines.queue(Instant::now());
        match tokio::time::timeout_at(deadline.into(), ticket.readmitted()).await {
            Ok(readmitted) => readmitted.map_err(Failure::Refused)?,
            Err(_) => return Err(Failure::Late(Phase::Queue, timeout)),
        }
        ex.resubmits += 1;
        request = http::Request::from_parts(again.0.clone(), again.1.again());
    }
}

/// Sends a request and reads its response to the first body frame, or to the end of a body without one: its headers by
/// the attempt's deadline, and its first frame by the request's.
async fn first_frame(
    upstream: &Upstream,
    request: http::Request<UpstreamBody>,
    deadlines: &Deadlines,
) -> Result<(http::Response<Incoming>, Option<Bytes>), Failure> {
    let (parts, mut body) = send(upstream, request, deadlines).await?.into_parts();
    let (deadline, timeout) = deadlines.first_frame();
    let first = async {
        loop {
            match body.frame().await {
                Some(Ok(frame)) => match frame.into_data() {
                    Ok(data) => break Ok(Some(data)),
                    Err(_) => continue,
                },
                Some(Err(_)) => break Err(Failure::Body),
                None => break Ok(None),
            }
        }
    };
    let first = match tokio::time::timeout_at(deadline.into(), first).await {
        Ok(first) => first?,
        Err(_) => return Err(Failure::Late(Phase::Upstream, timeout)),
    };
    Ok((http::Response::from_parts(parts, body), first))
}
