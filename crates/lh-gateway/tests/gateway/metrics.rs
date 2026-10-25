//! `/metrics` and `/metrics/engine`: the metrics key on the TCP listener, none on the private socket, and the
//! engine's exposition relayed.

use http::StatusCode;

use crate::common::*;

/// The JSON error of a request refused for its key.
fn unauthorized(reply: &Reply) {
    assert_eq!(reply.status, StatusCode::UNAUTHORIZED);
    assert_eq!(
        reply.json(),
        serde_json::json!({"error": {"message": "missing or unknown metrics key", "type": "authentication_error",
                                     "code": "invalid_api_key"}})
    );
}

#[tokio::test]
async fn metrics_need_the_metrics_key_on_tcp_and_none_on_the_private_socket() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    // Every user of the host reaches the TCP port: the tenants and their traffic stay behind the key, which no
    // tenant's key replaces.
    for headers in [
        &[][..],
        &[("authorization", "Bearer metrics-key2")],
        &[("authorization", "Basic bWV0cmljcy1rZXk=")],
        &[AGENTS_AUTH],
    ] {
        unauthorized(&call(&gw.tcp, "GET", "/metrics", headers, "").await);
    }
    let reply = call(&gw.tcp, "GET", "/metrics", &[METRICS_AUTH], "").await;
    assert_eq!(reply.status, 200);
    assert!(
        reply
            .header("content-type")
            .unwrap()
            .starts_with("application/openmetrics-text")
    );
    // The socket's file mode keeps other users out, as for its tenant's requests.
    let reply = call(&gw.unix, "GET", "/metrics", &[], "").await;
    assert_eq!(reply.status, 200);
    let text = String::from_utf8(reply.body.to_vec()).unwrap();
    assert!(text.contains("lh_gateway_request_outcomes_total"));
    // Liveness and readiness say nothing of the traffic and stay open.
    assert_eq!(call(&gw.tcp, "GET", "/health", &[], "").await.status, 200);
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 200);
    // The refusals are counted, as any request the gateway answers.
    let series = "lh_gateway_requests_total{tenant=\"\",route=\"other\",class=\"\",status=\"401\"}";
    gw.await_metric(series, 4.0).await;
}

#[tokio::test]
async fn the_engines_metrics_are_relayed_under_the_same_rule() {
    let mock = Mock::start_unix(Script::Json("{}")).await;
    let gw = Gateway::start_upstream(&mock.upstream, "").await;
    unauthorized(&call(&gw.tcp, "GET", "/metrics/engine", &[], "").await);
    for (target, headers) in [(&gw.tcp, &[METRICS_AUTH][..]), (&gw.unix, &[])] {
        let reply = call(target, "GET", "/metrics/engine", headers, "").await;
        assert_eq!(reply.status, 200);
        assert_eq!(reply.header("content-type"), Some(ENGINE_METRICS_TYPE));
        assert_eq!(&reply.body[..], ENGINE_METRICS.as_bytes());
    }
    // Only the path is the engine's: a request's headers do not reach it.
    assert!(mock.requests().is_empty());

    let gone = Gateway::start_upstream(&format!("unix:{}", socket_path().display()), "").await;
    let reply = call(&gone.tcp, "GET", "/metrics/engine", &[METRICS_AUTH], "").await;
    assert_eq!(reply.status, 502);
    assert_eq!(reply.json()["error"]["code"], "upstream_unavailable");
}
