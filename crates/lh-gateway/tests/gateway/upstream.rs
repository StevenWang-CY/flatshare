//! An upstream on a Unix socket, as the engine serves on its GPU node and the tunnel on the control host: requests and
//! streams, pooled connections and their idle timeout, cancellation, a socket nobody serves, and an engine that
//! restarts behind it.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::BodyExt;

use crate::common::*;

#[tokio::test]
async fn requests_and_streams_reach_an_engine_on_a_unix_socket() {
    let chunks: Vec<Bytes> = [
        &b"data: {\"choices\":[{\"delta\":{\"content\":\"h\"},\"finish_reason\":null}]}\n\n"[..],
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":1}}\n\n",
        b"data: [DONE]\n\n",
    ]
    .into_iter()
    .map(Bytes::from_static)
    .collect();
    let expected: Vec<u8> = chunks.concat();
    let mock = Mock::start_unix(Script::Sse {
        chunks,
        gap: Duration::from_millis(5),
    })
    .await;
    let gw = Gateway::start_upstream(&mock.upstream, "").await;
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 200);

    let headers = [("x-session-id", "voice-RM_a1B2c3")];
    let reply = call(&gw.unix, "POST", "/v1/chat/completions", &headers, CHAT).await;
    assert_eq!(reply.status, 200);
    assert_eq!(reply.header("content-type"), Some("text/event-stream"));
    assert_eq!(&reply.body[..], &expected[..]);
    let seen = mock.last();
    assert_eq!(seen.path, "/v1/chat/completions");
    assert_eq!(seen.header("x-vllm-priority"), Some("0"));
    assert_eq!(seen.header("x-session-id"), Some("voice-RM_a1B2c3"));
    assert_eq!(seen.json()["cache_salt"], VOICE_SALT);

    let models = call(&gw.tcp, "GET", "/v1/models", &[BENCH_AUTH], "").await;
    assert_eq!(models.status, 200);
    assert_eq!(mock.last().path, "/v1/models");
}

#[tokio::test]
async fn connections_to_a_unix_upstream_are_pooled_until_the_idle_timeout() {
    let mock = Mock::start_unix(Script::Json("{}")).await;
    let gw = Gateway::start_upstream(&mock.upstream, "upstream_idle_timeout_s = 0.3").await;
    let send = async || {
        let reply = call(
            &gw.tcp,
            "POST",
            "/v1/chat/completions",
            &[AGENTS_AUTH],
            CHAT,
        )
        .await;
        assert_eq!(reply.status, 200);
    };
    for _ in 0..3 {
        send().await;
    }
    let accepted = || mock.connections.load(Ordering::SeqCst);
    assert_eq!(
        accepted(),
        1,
        "one pooled connection carried the three requests"
    );
    // Idle past upstream_idle_timeout_s, the connection is let go and the next request opens another.
    tokio::time::sleep(Duration::from_millis(700)).await;
    send().await;
    assert_eq!(accepted(), 2);
}

#[tokio::test]
async fn a_client_that_leaves_mid_stream_closes_the_unix_upstream_before_its_next_chunk() {
    // The engine sends three chunks, then nothing: only the client's leaving can end the stream.
    let chunk = Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
    let mut mock = Mock::start_unix(Script::Quiet {
        chunks: vec![chunk; 3],
    })
    .await;
    let gw = Gateway::start_upstream(&mock.upstream, "").await;
    let (response, connection) = open(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    let mut body = response.into_body();
    for _ in 0..3 {
        body.frame().await.unwrap().unwrap();
    }
    connection.abort();
    drop(body);
    mock.next_drop(Duration::from_secs(2)).await;
    gw.await_metric(
        "lh_gateway_request_outcomes_total{class=\"background\",outcome=\"cancelled\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn a_unix_upstream_nobody_serves_is_502_and_not_ready() {
    // A socket file left by an engine that died, and a path with no file at all.
    let stale = socket_path();
    drop(std::os::unix::net::UnixListener::bind(&stale).unwrap());
    for (path, cause) in [
        (stale, "Connection refused"),
        (socket_path(), "No such file or directory"),
    ] {
        let gw = Gateway::start_upstream(&format!("unix:{}", path.display()), "").await;
        let reply = call(
            &gw.tcp,
            "POST",
            "/v1/chat/completions",
            &[AGENTS_AUTH],
            CHAT,
        )
        .await;
        assert_eq!(reply.status, 502);
        let error = &reply.json()["error"];
        assert_eq!(error["code"], "upstream_unavailable");
        let message = error["message"].as_str().unwrap();
        assert!(message.contains(cause), "{message}");
        gw.await_metric("lh_gateway_upstream_errors_total{kind=\"connect\"}", 1.0)
            .await;
        assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 503);
    }
}

#[tokio::test]
async fn while_the_engine_restarts_requests_are_503_with_retry_after_until_it_answers() {
    // The engine's socket, which nobody serves until the engine is back.
    let path = socket_path();
    let gw = Gateway::start_upstream(&format!("unix:{}", path.display()), "").await;
    let chat = || {
        call(
            &gw.tcp,
            "POST",
            "/v1/chat/completions",
            &[AGENTS_AUTH],
            CHAT,
        )
    };
    // The first request finds the engine gone, and starts the probe of its /health.
    let first = chat().await;
    assert_eq!(first.status, 502);
    assert_eq!(first.json()["error"]["code"], "upstream_unavailable");
    gw.await_metric("lh_gateway_node_ready", 0.0).await;
    // Until the probe answers, requests are answered at once without going upstream, and told when to come back.
    let started = Instant::now();
    let refused = chat().await;
    assert!(started.elapsed() < Duration::from_millis(100));
    assert_eq!(refused.status, 503);
    let error = &refused.json()["error"];
    assert_eq!(error["type"], "server_error");
    assert_eq!(error["code"], "upstream_not_ready");
    assert_eq!(refused.header("retry-after"), Some("1"));
    let models = call(&gw.tcp, "GET", "/v1/models", &[AGENTS_AUTH], "").await;
    assert_eq!(
        (models.status.as_u16(), models.header("retry-after")),
        (503, Some("1"))
    );
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 503);
    let connects = "lh_gateway_upstream_errors_total{kind=\"connect\"}";
    assert_eq!(metric(&gw.metrics().await, connects), Some(1.0));
    // The engine is back on its socket: once the probe answers, requests go upstream again.
    let engine = Mock::start_unix_at(path, Script::Json("{}")).await;
    gw.await_metric("lh_gateway_node_ready", 1.0).await;
    assert_eq!(chat().await.status, 200);
    assert_eq!(engine.requests().len(), 1);
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 200);
}

#[tokio::test]
async fn a_connection_that_breaks_on_a_serving_engine_makes_no_request_wait() {
    let mock = Mock::start_flaky(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    let chat = || {
        call(
            &gw.tcp,
            "POST",
            "/v1/chat/completions",
            &[AGENTS_AUTH],
            CHAT,
        )
    };
    assert_eq!(chat().await.status, 502);
    // The probe that the failure started finds the engine serving: the next request goes upstream at once.
    assert_eq!(chat().await.status, 200);
    assert_eq!(
        metric(&gw.metrics().await, "lh_gateway_node_ready"),
        Some(1.0)
    );
}
