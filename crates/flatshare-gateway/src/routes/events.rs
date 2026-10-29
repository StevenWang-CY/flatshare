//! `POST /v1/events`: lifecycle events from trusted tenants, recorded by [`crate::events`].

use axum::extract::{Request, State};
use axum::response::Response;
use http::StatusCode;

use super::{read_control_body, trusted_tenant};
use crate::events::parse;
use crate::exchange::Exchange;
use crate::state::App;
use crate::types::Route;

/// `POST /v1/events`: one event object or an array of them. Answers 202 `{"accepted": n, "duplicates": m}`.
/// A batch is validated in full before any of it is recorded.
pub(crate) async fn post(State(app): State<App>, request: Request) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, Route::Events);
    let action = "post lifecycle events";
    let tenant = match trusted_tenant(&app, &mut ex, request.headers(), action) {
        Ok(tenant) => tenant,
        Err((status, code, message)) => return ex.error(status, code, message),
    };
    // A batch's bytes come from the budget of the bodies the gateway holds, as a request's do, and it has the
    // interactive class's upload deadline.
    let mut budget = shared.intake.body();
    let (body, again) = (request.into_body(), "send the events again");
    let read = read_control_body(shared, ex.received, body, &mut budget, "events", again);
    let body = match read.await {
        Ok(body) => body,
        Err((status, code, message)) => return ex.error(status, code, message),
    };
    let events = match parse(&body) {
        Ok(events) => events,
        Err(message) => return ex.error(StatusCode::BAD_REQUEST, "invalid_event", message),
    };
    let (accepted, duplicates) =
        shared
            .events
            .ingest(&tenant.name, events, &shared.metrics, |fresh| {
                shared.lifecycle(tenant, fresh)
            });
    let body = serde_json::json!({"accepted": accepted, "duplicates": duplicates});
    ex.respond(StatusCode::ACCEPTED, &body)
}
