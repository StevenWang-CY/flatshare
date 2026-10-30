//! The proxy path: authenticate, sanitize, resolve, admit, and send the request upstream. The engine and its
//! readiness are [`crate::upstream`]'s, the nodes and their placements [`crate::nodes`]', and the response's relay to
//! the client [`crate::relay`]'s.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::Request;
use axum::response::Response;
use bytes::Bytes;
use bytes::BytesMut;
use flatshare_proto::{SessionId, SloClass};
use http::header::{CONTENT_TYPE, HeaderValue};
use http::uri::PathAndQuery;
use http::{StatusCode, Version};
use http_body::Body as _;
use http_body_util::BodyExt;
use hyper::body::Incoming;

use crate::auth::Tenant;
use crate::deadline::{Deadlines, Phase, Timeout};
use crate::drain;
use crate::engine::Engine;
use crate::exchange::Exchange;
use crate::harness::{Refusal, Ticket};
use crate::intake::{BodyBytes, Refused};
use crate::nodes::{Key, Nodes, Placed, Unplaced};
use crate::relay;
use crate::sanitize::{
    self, Bounds, InvalidBody, REQUEST_ID, cap_reply, names_session, resolve_class,
    resolve_session, rewrite_body, sanitize_headers, set_engine_headers, strip_hop_by_hop,
};
use crate::state::{App, Engines, Shared};
use crate::types::{Route, UpstreamError};
use crate::upstream::{Upstream, UpstreamBody, causes};

/// Where a request goes.
pub(crate) enum Target {
    /// An engine: the gateway's one upstream, or a node's for a request without a session (a model list).
    Engine(Arc<Engine>),
    /// A request placed on a node, whose engine it goes to.
    Placed(Placed),
}

/// Why a request has nowhere to go.
enum NoTarget {
    /// The gateway's one engine is not ready.
    NotReady,
    /// No node takes the request's session.
    Unplaced(Unplaced),
}

impl Target {
    /// Where `tenant`'s request of `class` and `session` goes: the one engine while it is ready, or with `[[nodes]]` the
    /// session's node, or a node chosen for it. A request of a session the gateway minted (`named` false) is placed
    /// alone.
    fn for_request(
        shared: &Shared,
        tenant: &Tenant,
        class: SloClass,
        session: &SessionId,
        named: bool,
    ) -> Result<Self, NoTarget> {
        match &shared.engines {
            Engines::One(engine) => {
                if !engine.upstream.ready() {
                    return Err(NoTarget::NotReady);
                }
                Ok(Self::Engine(Arc::clone(engine)))
            }
            Engines::Nodes(nodes) => {
                let key = named.then(|| Key {
                    tenant: tenant.index,
                    session: session.clone(),
                });
                nodes
                    .request(key, class, tenant.trusted)
                    .map(Self::Placed)
                    .map_err(NoTarget::Unplaced)
            }
        }
    }
}

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
    ex.resolved.tenant = Some(Arc::clone(&tenant.name));

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
    let deadlines = ex.classify(class, shared.class(class));
    if let Engines::One(engine) = &shared.engines {
        ex.routed.health = Some(Arc::clone(&engine.health));
    }

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
    let mut body = match rewrite(shared, tenant, class, route, &body) {
        Ok(body) => body,
        Err(invalid) => return ex.invalid_body(&invalid),
    };
    let named = names_session(&inputs, body.prompt_cache_key.as_deref());
    let session = resolve_session(tenant, &inputs, body.prompt_cache_key.as_deref());
    let signals = shared.upstream_priority;
    set_engine_headers(&mut parts.headers, &session, class, &ex.id, signals);
    parts
        .headers
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let session = &*ex.resolved.session.insert(session);
    let mut target = match Target::for_request(shared, tenant, class, session, named) {
        Ok(target) => target,
        Err(NoTarget::NotReady) => return ex.engine_not_ready(),
        Err(NoTarget::Unplaced(unplaced)) => return ex.unplaced(unplaced, class),
    };

    // The admission point, under an [admission] policy. Its decisions are the gateway's overhead; the time a request
    // waits there is not.
    let mut start = read;
    let ticket = match &shared.admission {
        None => None,
        Some(admission) => {
            let arrived = Instant::now();
            let (queue, len) = (deadlines.queue(arrived), body.json.len());
            match admission
                .wait(tenant, class, session, len, queue, &shared.drain)
                .await
            {
                Ok(ticket) => {
                    start += ex.admitted_after(ticket.as_ref());
                    ticket
                }
                Err(why) => return ex.not_admitted(why, arrived.elapsed(), shared.drain.grace),
            }
        }
    };

    // A node that covers holds its agents' requests back; the time they wait is not the overhead.
    if let (Target::Placed(placed), Some(nodes)) = (&mut target, shared.nodes()) {
        let arrived = Instant::now();
        match nodes
            .hold(placed, deadlines.queue(arrived), &shared.drain)
            .await
        {
            Ok(waited) => start += ex.held_back(waited),
            Err(why) => return ex.not_released(why, arrived.elapsed(), shared.drain.grace),
        }
    }

    // Degrade's reply cap, over the class's: the lower of the two holds.
    if let Some(cap) = ticket.as_ref().and_then(Ticket::reply_cap) {
        body.json = cap_reply(&body.json, route, cap);
        ex.harness.degraded = true;
    }

    let body = UpstreamBody::new(body.json, start, shared.metrics.overhead(route));
    forward(&app, ex, parts, body, ticket, target).await
}

/// The body as the engine is to get it, in the rewrite's one pass ([`rewrite_body`]): the tenant's cache salt, the
/// class's priority where the upstream reads it from the body, and the class's reply cap and the bound on choices.
fn rewrite(
    shared: &Shared,
    tenant: &Tenant,
    class: SloClass,
    route: Route,
    body: &[u8],
) -> Result<sanitize::Body, InvalidBody> {
    let priority = shared
        .upstream_priority
        .body
        .then(|| class.engine_priority());
    let bounds = Bounds {
        route,
        cap: shared.class(class).max_reply_tokens,
        max_choices: shared.max_choices,
    };
    rewrite_body(body, &tenant.cache_salt, priority, bounds, &shared.metrics)
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
    ex.resolved.tenant = Some(Arc::clone(&tenant.name));
    // A model list has no class, and the interactive class's deadlines.
    let interactive = SloClass::Interactive;
    ex.resolved.deadlines = Some(Deadlines::new(
        interactive,
        shared.class(interactive),
        ex.received,
    ));
    sanitize_headers(&mut parts.headers, tenant.trusted, &shared.metrics);
    parts.headers.insert(REQUEST_ID.clone(), ex.id.clone());
    let engine = match &shared.engines {
        Engines::One(engine) => {
            if !engine.upstream.ready() {
                return ex.engine_not_ready();
            }
            Arc::clone(engine)
        }
        // Any available node's: the nodes serve one model.
        Engines::Nodes(nodes) => match nodes.any() {
            Some(engine) => engine,
            None => {
                let message = "no node is healthy; ask again";
                return ex.not_ready("no_healthy_node", message, Duration::from_secs(1));
            }
        },
    };
    let target = Target::Engine(engine);
    forward(&app, ex, parts, UpstreamBody::empty(), None, target).await
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
/// first byte is sent again; a realtime one whose node fails before its first content is sent once more on another
/// node ([`retrying`]).
async fn forward(
    app: &App,
    mut ex: Exchange,
    mut parts: http::request::Parts,
    body: UpstreamBody,
    mut ticket: Option<Ticket>,
    target: Target,
) -> Response {
    let shared = &app.shared;
    let (engine, retry) = match target {
        Target::Engine(engine) => (engine, None),
        Target::Placed(placed) => {
            let engine = Arc::clone(&placed.node.engine);
            ex.place_on(placed);
            // Only a realtime request is sent again, and only where another node could take it.
            let realtime = ex.resolved.class == Some(SloClass::Realtime);
            let retry = shared
                .nodes()
                .filter(|nodes| realtime && nodes.list.len() > 1);
            (engine, retry.cloned())
        }
    };
    let upstream = &engine.upstream;
    let path = parts
        .uri
        .path_and_query()
        .cloned()
        .unwrap_or_else(|| PathAndQuery::from_static("/"));
    parts.uri = upstream.uri(path.clone());
    parts.version = Version::HTTP_11;
    parts.extensions.clear();

    ex.admit(ticket.is_some());
    let deadlines = ex
        .resolved
        .deadlines
        .expect("a forwarded request has its deadlines");
    // A background request without a response at the drain's cut is dropped, as is a shed one or one whose deadline
    // passes: dropping the request future closes its connection, which aborts the engine request.
    let mut cut = shared.drain.cut_for(ex.resolved.class);
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
            _ => match &retry {
                Some(nodes) => retrying(nodes, &mut ex, parts, body, path, &deadlines).await,
                None => send(upstream, http::Request::from_parts(parts, body), &deadlines)
                    .await
                    .map(|response| (response, None)),
            },
        }
    };
    let sent = tokio::select! {
        biased;
        () = drain::cut(&mut cut) => Err(Failure::Drained),
        sent = sending => sent,
    };
    let (response, first) = match sent {
        Ok(sent) => sent,
        Err(failure) => {
            // An engine that a request found broken is probed until it answers; meanwhile its requests are
            // answered 503, or a node's placed on another node ([`crate::upstream::recover`]). After a retry, the
            // engine is the second node's.
            if failure.broken() {
                match &ex.routed.placed {
                    Some(placed) => placed.node.engine.found_broken(),
                    None => engine.found_broken(),
                }
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
        ex.harness.resubmits += 1;
        request = http::Request::from_parts(again.0.clone(), again.1.again());
    }
}

/// Sends a realtime request placed on a node and holds its response back until its first body frame. If its node
/// fails before that frame (the connection fails, or closes before the response headers, or the body breaks off before
/// its first frame), the attempt is dropped, which closes its connection and aborts the engine's work, the node's
/// engine is probed, and the request is sent once more on another available node, where its session moves, within the
/// deadlines of its receipt. The client has seen nothing of the first attempt. A request whose first content has
/// reached the client is never sent again: its relay ends it with an error event, so that a voice reply is never a
/// second generation appended to half-delivered audio. A slow engine is no failure: a deadline ends the request.
async fn retrying(
    nodes: &Arc<Nodes>,
    ex: &mut Exchange,
    parts: http::request::Parts,
    body: UpstreamBody,
    path: PathAndQuery,
    deadlines: &Deadlines,
) -> Result<(http::Response<Incoming>, Option<Bytes>), Failure> {
    let again = (parts.clone(), body.again());
    let placed = ex.routed.placed.as_ref().expect("a placed request");
    let engine = Arc::clone(&placed.node.engine);
    let request = http::Request::from_parts(parts, body);
    let failure = match first_frame(&engine.upstream, request, deadlines).await {
        Ok(sent) => return Ok(first_content(ex, sent)),
        Err(failure) if failure.broken() => failure,
        Err(failure) => return Err(failure),
    };
    engine.found_broken();
    // No other node takes it: the client gets the failure.
    let Ok(moved) = nodes.retry(placed) else {
        return Err(failure);
    };
    let engine = Arc::clone(&moved.node.engine);
    // The first attempt's place on its node goes.
    ex.place_on(moved);
    ex.routed.retries += 1;
    let (mut parts, body) = again;
    parts.uri = engine.upstream.uri(path);
    let request = http::Request::from_parts(parts, body);
    let sent = first_frame(&engine.upstream, request, deadlines).await?;
    Ok(first_content(ex, sent))
}

/// Records the first frame of a response read before the client was answered: its first byte, and for a 2xx its
/// engine's success.
fn first_content(
    ex: &mut Exchange,
    (response, first): (http::Response<Incoming>, Option<Bytes>),
) -> (http::Response<Incoming>, Option<Bytes>) {
    if first.is_some() {
        ex.first_byte = Some(ex.received.elapsed());
        if response.status().is_success() {
            ex.succeeded();
        }
    }
    (response, first)
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
