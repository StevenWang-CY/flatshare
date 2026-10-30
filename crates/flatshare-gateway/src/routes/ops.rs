//! The operator's routes: `GET /metrics`, `GET /metrics/engine` and `GET /ready`.

use std::time::Duration;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use http::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use http::{StatusCode, Uri};

use super::metrics_refused;
use crate::exchange::Exchange;
use crate::state::{App, Engines};
use crate::types::{EngineState, Route};

/// `GET /ready` succeeds while the upstream's `/health` has answered within this window.
const READY_WINDOW: Duration = Duration::from_secs(1);

/// How long `GET /metrics/engine` waits for the engine's `/metrics`: Prometheus' default scrape timeout.
const ENGINE_METRICS_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) async fn metrics(State(app): State<App>, headers: HeaderMap) -> Response {
    if let Some(refused) = metrics_refused(&app, &headers) {
        return refused;
    }
    let content_type =
        HeaderValue::from_static("application/openmetrics-text; version=1.0.0; charset=utf-8");
    ([(CONTENT_TYPE, content_type)], app.shared.metrics.encode()).into_response()
}

/// `GET /metrics/engine`: the engine's own `/metrics`, under the rule of the gateway's, so that a scraper reaches
/// the engine without a TCP route to it. With `[[nodes]]`, `?node=NAME` names the node whose engine's.
pub(crate) async fn engine_metrics(
    State(app): State<App>,
    headers: HeaderMap,
    uri: Uri,
) -> Response {
    if let Some(refused) = metrics_refused(&app, &headers) {
        return refused;
    }
    let shared = &app.shared;
    let error = |status, code, message: String| {
        Exchange::new(&shared.metrics, Route::Other).error(status, code, message)
    };
    let upstream = match &shared.engines {
        Engines::One(engine) => &engine.upstream,
        Engines::Nodes(nodes) => {
            let asked = uri
                .query()
                .and_then(|query| query.split('&').find_map(|pair| pair.strip_prefix("node=")));
            let names = || {
                let names: Vec<&str> = nodes.list.iter().map(|node| &*node.name).collect();
                names.join(", ")
            };
            match asked {
                None => {
                    let message = format!("name a node: /metrics/engine?node=NAME, of {}", names());
                    return error(StatusCode::BAD_REQUEST, "node_required", message);
                }
                Some(name) => match nodes.named(name) {
                    Some(node) => &node.engine.upstream,
                    None => {
                        let message = format!("no node {name:?}: the nodes are {}", names());
                        return error(StatusCode::NOT_FOUND, "unknown_node", message);
                    }
                },
            }
        }
    };
    match upstream.metrics(ENGINE_METRICS_TIMEOUT).await {
        Ok(response) => response,
        Err(message) => Exchange::new(&shared.metrics, Route::Other).error(
            StatusCode::BAD_GATEWAY,
            "upstream_unavailable",
            message,
        ),
    }
}

/// 200 while the engine serves ([`crate::canary`]: an inference succeeded, and the canary has not failed since
/// `degraded_after` times in a row; from the start without a canary) and its `/health` has answered within
/// [`READY_WINDOW`], asking it once more otherwise. 503 while it is warming, degraded or drained, and at once while a
/// request has found the upstream broken and its probe has not answered since ([`crate::upstream::recover`]). With
/// `[[nodes]]`, 200 while each role is taken by an available node, and 503 naming the role that is not, or once the
/// gateway drains.
pub(crate) async fn ready(State(app): State<App>) -> Response {
    let shared = &app.shared;
    let engine = match &shared.engines {
        Engines::One(engine) => engine,
        // Every role taken by an available node: healthy, and its engine ready.
        Engines::Nodes(nodes) => {
            if nodes
                .list
                .iter()
                .any(|node| node.engine.health.state() == EngineState::Drained)
            {
                return (StatusCode::SERVICE_UNAVAILABLE, "gateway drained\n").into_response();
            }
            return match nodes.ready() {
                Ok(()) => (StatusCode::OK, "ready\n").into_response(),
                Err(role) => {
                    let body = format!("no healthy node for {role}\n");
                    (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
                }
            };
        }
    };
    let state = engine.health.state();
    if state != EngineState::Serving {
        let body = format!("engine {}\n", state.as_str());
        return (StatusCode::SERVICE_UNAVAILABLE, body).into_response();
    }
    let upstream = &engine.upstream;
    if upstream.ready() && upstream.answers(READY_WINDOW).await {
        (StatusCode::OK, "ready\n").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "upstream not ready\n").into_response()
    }
}
