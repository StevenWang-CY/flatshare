//! The gateway's HTTP routes: which handler answers each path, and what the control routes share. The inference
//! routes' handlers are the proxy's ([`crate::proxy`]); the lifecycle events' ([`events`]), the nodes' ([`nodes`])
//! and the operator's ([`ops`]) are here.

mod events;
mod nodes;
mod ops;

use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::body::Body;
use axum::extract::rejection::PathRejection;
use axum::extract::{Path, Request, State};
use axum::response::Response;
use axum::routing::{MethodRouter, delete, get, post};
use bytes::Bytes;
use flatshare_proto::SloClass;
use http::header::HeaderMap;
use http::{Method, StatusCode, Uri};

use crate::auth::{self, Tenant};
use crate::deadline::Deadlines;
use crate::exchange::{self, ErrorAnswer, Exchange};
use crate::intake::BodyBytes;
use crate::nodes::Nodes;
use crate::proxy::{self, BodyError, read_body};
use crate::state::{App, Shared};
use crate::types::Route;

/// The routes of one listener's requests.
pub(crate) fn router(app: App) -> Router {
    let router = Router::new();
    // Without [[nodes]] none of these routes exists: their paths are 404, as they always were.
    let router = match app.shared.nodes() {
        Some(nodes) => nodes_routes(router, nodes),
        None => router,
    };
    router
        .route("/v1/chat/completions", inference(Route::ChatCompletions))
        .route("/v1/completions", inference(Route::Completions))
        .route("/v1/responses", inference(Route::Responses))
        .route("/v1/messages", inference(Route::Messages))
        .route("/v1/messages/count_tokens", inference(Route::CountTokens))
        .route(
            "/v1/models",
            get(|State(app): State<App>, request: Request| proxy::models(app, request)),
        )
        .route("/v1/events", post(events::post))
        .route("/metrics", get(ops::metrics))
        .route("/metrics/engine", get(ops::engine_metrics))
        .route("/health", get(|| async { "ok\n" }))
        .route("/ready", get(ops::ready))
        .fallback(
            |State(app): State<App>, method: Method, uri: Uri| async move {
                Exchange::new(&app.shared.metrics, Route::Other).error(
                    StatusCode::NOT_FOUND,
                    "not_found",
                    format!("no such route: {}", request_line(&method, &uri)),
                )
            },
        )
        .method_not_allowed_fallback(
            |State(app): State<App>, method: Method, uri: Uri| async move {
                Exchange::new(&app.shared.metrics, Route::Other).error(
                    StatusCode::METHOD_NOT_ALLOWED,
                    "method_not_allowed",
                    format!(
                        "the route does not accept this method: {}",
                        request_line(&method, &uri)
                    ),
                )
            },
        )
        .with_state(app)
}

/// The routes a gateway with `[[nodes]]` adds, each handed the nodes.
fn nodes_routes(router: Router<App>, nodes: &Arc<Nodes>) -> Router<App> {
    let (sessions, end, list) = (Arc::clone(nodes), Arc::clone(nodes), Arc::clone(nodes));
    router
        .route(
            "/v1/sessions",
            post(move |State(app): State<App>, request: Request| async move {
                nodes::sessions(app, &sessions, request).await
            }),
        )
        .route(
            "/v1/sessions/{id}",
            delete(
                move |State(app): State<App>,
                      id: Result<Path<String>, PathRejection>,
                      headers: HeaderMap| async move {
                    nodes::end(app, &end, id, headers).await
                },
            ),
        )
        .route(
            "/nodes",
            get(move |State(app): State<App>, request: Request| async move {
                nodes::nodes(app, &list, request).await
            }),
        )
}

/// The method and path of a request that matched no route, for its error and log line. The `route` label of
/// such a request is "other", so this is the only record of what was asked for; long paths are cut.
fn request_line(method: &Method, uri: &Uri) -> String {
    const MAX_PATH: usize = 256;
    let path = uri.path();
    let end = (0..=MAX_PATH.min(path.len()))
        .rev()
        .find(|&i| path.is_char_boundary(i))
        .unwrap_or(0);
    let cut = if end < path.len() { "…" } else { "" };
    format!("{method} {}{cut}", &path[..end])
}

fn inference(route: Route) -> MethodRouter<App> {
    post(move |State(app): State<App>, request: Request| proxy::inference(app, route, request))
}

/// The tenant of a control route's request, which only a trusted tenant may make, recorded in `ex`. Or the error
/// answer: 401 without a known API key on a listener without a tenant ([`exchange::missing_key`]), and 403 for an
/// untrusted tenant, which may not `action`.
pub(crate) fn trusted_tenant<'a>(
    app: &'a App,
    ex: &mut Exchange,
    headers: &HeaderMap,
    action: &str,
) -> Result<&'a Tenant, ErrorAnswer> {
    let Some(tenant) = app
        .shared
        .tenants
        .authenticate(headers, app.listener_tenant)
    else {
        return Err(exchange::missing_key());
    };
    ex.resolved.tenant = Some(Arc::clone(&tenant.name));
    if !tenant.trusted {
        let message = format!("only trusted tenants may {action}");
        return Err((StatusCode::FORBIDDEN, "untrusted_tenant", message));
    }
    Ok(tenant)
}

/// Reads a control route's body as a request's is read: its bytes from `budget`, the bodies' budget, as they arrive,
/// by the interactive class's upload deadline from `received`. The error is the answer's status, code and message:
/// 408 when the body, `what`, did not arrive in time, 413 or 400 for one too large or unreadable, and 503 when the
/// budget is full, which says what to do `again`.
pub(crate) async fn read_control_body(
    shared: &Shared,
    received: Instant,
    body: Body,
    budget: &mut BodyBytes,
    what: &str,
    again: &str,
) -> Result<Bytes, ErrorAnswer> {
    let interactive = SloClass::Interactive;
    let upload = Deadlines::new(interactive, shared.class(interactive), received).upload();
    read_body(body, shared.max_body_bytes, budget, upload)
        .await
        .map_err(|err| match err {
            BodyError::Late(timeout) => {
                let message = format!("the {what} did not arrive within {}", timeout.key());
                (StatusCode::REQUEST_TIMEOUT, "upload_timeout", message)
            }
            BodyError::Invalid(status, code, message) => (status, code, message),
            BodyError::Refused(_) => {
                let message =
                    format!("the gateway holds max_buffered_body_bytes of request bodies; {again}");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "body_budget_exhausted",
                    message,
                )
            }
        })
}

/// The 401 of a request that may not read the metrics: one on a TCP listener without the metrics key. On a Unix
/// socket, private to its owner, no key is asked for, as for the requests of its tenant.
pub(crate) fn metrics_refused(app: &App, headers: &HeaderMap) -> Option<Response> {
    let key = app.shared.metrics_key.as_ref();
    let allowed = app.private || key.is_some_and(|key| auth::is_metrics_key(headers, key));
    (!allowed).then(|| {
        Exchange::new(&app.shared.metrics, Route::Other).error(
            StatusCode::UNAUTHORIZED,
            "invalid_api_key",
            "missing or unknown metrics key",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_lines_name_the_method_and_cut_long_paths() {
        let uri: Uri = "/prometheus/metrics?x=1".parse().unwrap();
        assert_eq!(request_line(&Method::GET, &uri), "GET /prometheus/metrics");
        let long: Uri = format!("/{}", "a".repeat(300)).parse().unwrap();
        let line = request_line(&Method::POST, &long);
        assert_eq!(line, format!("POST /{}…", "a".repeat(255)));
    }
}
