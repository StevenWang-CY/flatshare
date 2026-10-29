//! The routes a gateway with `[[nodes]]` adds: `POST /v1/sessions`, with which the voice worker places a call
//! before it speaks and learns which node's speech servers to bind, `DELETE /v1/sessions/{id}`, which ends a session's
//! placement, and `GET /nodes`, the nodes as the gateway judges them. Without `[[nodes]]` none exists (404).

use std::sync::Arc;

use axum::extract::rejection::PathRejection;
use axum::extract::{Path, Request, State};
use axum::response::{IntoResponse, Response};
use flatshare_proto::{SessionId, SloClass};
use http::StatusCode;
use http::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;

use super::Key;
use crate::deadline::Deadlines;
use crate::exchange::Exchange;
use crate::metrics::Route;
use crate::proxy::{BodyError, read_body};
use crate::server::{App, metrics_refused};

/// The body of `POST /v1/sessions`; unknown fields are ignored.
#[derive(Deserialize)]
struct Asked {
    session_id: SessionId,
    /// The tenant's `default_class` when absent.
    class: Option<SloClass>,
}

/// `POST /v1/sessions` from a trusted tenant: `{"session_id": ..., "class": ...}` gets 200 `{"session_id", "class",
/// "node", "stt", "tts"}`, the node the session is placed on (placed now if it has none, or if its node is not
/// available: a move) and that node's speech servers as configured, `null` for one it has not. Asking again
/// answers the same node while the placement holds, so a call that asks at each turn learns of a move.
pub(crate) async fn sessions(State(app): State<App>, request: Request) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, Route::Sessions);
    let Some(nodes) = shared.nodes() else {
        return ex.error(
            StatusCode::NOT_FOUND,
            "not_found",
            "no such route: POST /v1/sessions",
        );
    };
    let (parts, body) = request.into_parts();
    let Some(tenant) = shared
        .tenants
        .authenticate(&parts.headers, app.listener_tenant)
    else {
        return ex.unauthorized();
    };
    ex.tenant = Some(Arc::clone(&tenant.name));
    if !tenant.trusted {
        return ex.error(
            StatusCode::FORBIDDEN,
            "untrusted_tenant",
            "only trusted tenants may place sessions",
        );
    }
    // As a batch of events: the bodies' budget, and the interactive class's upload deadline.
    let mut budget = shared.intake.body();
    let interactive = SloClass::Interactive;
    let upload = Deadlines::new(interactive, shared.class(interactive), ex.received).upload();
    let body = match read_body(body, shared.max_body_bytes, &mut budget, upload).await {
        Ok(body) => body,
        Err(BodyError::Late(timeout)) => {
            let message = format!("the session did not arrive within {}", timeout.key());
            return ex.error(StatusCode::REQUEST_TIMEOUT, "upload_timeout", message);
        }
        Err(BodyError::Invalid(status, code, message)) => return ex.error(status, code, message),
        Err(BodyError::Refused(_)) => {
            return ex.error(
                StatusCode::SERVICE_UNAVAILABLE,
                "body_budget_exhausted",
                "the gateway holds max_buffered_body_bytes of request bodies; ask again",
            );
        }
    };
    let asked: Asked = match serde_json::from_slice(&body) {
        Ok(asked) => asked,
        Err(err) => {
            let message = format!("expected {{\"session_id\": ..., \"class\": ...}}: {err}");
            return ex.error(StatusCode::BAD_REQUEST, "invalid_session", message);
        }
    };
    // As for a request, a class stricter than the tenant's max_class is clamped to it.
    let wanted = asked.class.unwrap_or(tenant.default_class);
    let class = wanted.max(tenant.max_class);
    if class != wanted {
        shared.metrics.class_clamped();
    }
    ex.class = Some(class);
    let key = Key {
        tenant: tenant.index,
        session: asked.session_id.clone(),
    };
    ex.session = Some(asked.session_id);
    let node = match nodes.session(&key, class, tenant.trusted) {
        Ok(node) => node,
        Err(unplaced) => return ex.unplaced(unplaced, class),
    };
    ex.node = Some(Arc::clone(&node.name));
    let address = |speech: &Option<super::Speech>| speech.as_ref().map(|s| s.addr.to_string());
    let body = serde_json::json!({
        "session_id": key.session.as_str(),
        "class": class,
        "node": &*node.name,
        "stt": address(&node.stt),
        "tts": address(&node.tts),
    });
    ex.respond(StatusCode::OK, &body)
}

/// `DELETE /v1/sessions/{id}` from a trusted tenant: ends the placement of its session `id`, which frees the session's
/// call or agent slot at once, as its `session_closed` event or its idle expiry does, and answers 200 `{"session_id",
/// "ended"}`, `ended` false for a session that had none (never placed, or ended already), so that ending a session
/// twice, or after its `session_closed`, succeeds.
pub(crate) async fn end(
    State(app): State<App>,
    id: Result<Path<String>, PathRejection>,
    headers: HeaderMap,
) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, Route::Session);
    let Some(nodes) = shared.nodes() else {
        return ex.error(
            StatusCode::NOT_FOUND,
            "not_found",
            "no such route: DELETE /v1/sessions/{id}",
        );
    };
    let Some(tenant) = shared.tenants.authenticate(&headers, app.listener_tenant) else {
        return ex.unauthorized();
    };
    ex.tenant = Some(Arc::clone(&tenant.name));
    if !tenant.trusted {
        return ex.error(
            StatusCode::FORBIDDEN,
            "untrusted_tenant",
            "only trusted tenants may end sessions",
        );
    }
    let session = id
        .map_err(|rejection| rejection.body_text())
        .and_then(|Path(id)| SessionId::new(id).map_err(|err| err.to_string()));
    let key = match session {
        Ok(session) => Key {
            tenant: tenant.index,
            session,
        },
        Err(err) => {
            let message = format!("expected a session id in the path: {err}");
            return ex.error(StatusCode::BAD_REQUEST, "invalid_session", message);
        }
    };
    let ended = nodes.end(&key);
    ex.node = ended.map(|node| Arc::clone(&node.name));
    let body = serde_json::json!({"session_id": key.session.as_str(), "ended": ex.node.is_some()});
    ex.session = Some(key.session);
    ex.respond(StatusCode::OK, &body)
}

/// `GET /nodes`: each node's verdict, its engine's state and readiness, each component's last check, its roles, call
/// slots and agents' bound, whether it covers (holds its agents back for calls moved to it or a warm spare's), and
/// its sessions and requests in flight by class; and how calls are placed and a covering node's agents held. It reads the metrics' rule: the metrics key on a TCP listener,
/// nothing on a Unix socket.
pub(crate) async fn nodes(State(app): State<App>, request: Request) -> Response {
    if let Some(refused) = metrics_refused(&app, request.headers()) {
        return refused;
    }
    let Some(nodes) = app.shared.nodes() else {
        return Exchange::new(&app.shared.metrics, Route::Other).error(
            StatusCode::NOT_FOUND,
            "not_found",
            "no such route: GET /nodes",
        );
    };
    let content_type = HeaderValue::from_static("application/json");
    let mut body = nodes.describe().to_string();
    body.push('\n');
    ([(CONTENT_TYPE, content_type)], body).into_response()
}
