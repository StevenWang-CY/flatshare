//! The routes a gateway with `[[nodes]]` adds: `POST /v1/sessions`, with which the voice worker places a call
//! before it speaks and learns which node's speech servers to bind, and `GET /nodes`, the nodes as the gateway judges
//! them. Without `[[nodes]]` neither exists (404).

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use flatshare_proto::{SessionId, SloClass};
use http::StatusCode;
use http::header::{CONTENT_TYPE, HeaderValue};
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

/// `GET /nodes`: each node's verdict, its engine's state and readiness, each component's last check, its roles, call
/// slots and agents' bound, whether it covers (holds its agents back for calls moved to it or a warm spare's), and
/// its sessions and requests in flight by class. It reads the metrics' rule: the metrics key on a TCP listener,
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
