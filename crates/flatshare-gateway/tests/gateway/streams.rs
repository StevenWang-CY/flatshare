//! Relaying: cancellation, backpressure, upstream failures, shutdown, and the Unix socket listener.

use std::io;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use flatshare_gateway::{Config, LogSink, Server};
use http_body_util::BodyExt;
use tokio::net::{TcpListener, TcpStream};

use crate::common::*;

const OUTCOME: &str = "flatshare_gateway_request_outcomes_total{class=\"background\",outcome=";

#[tokio::test]
async fn a_client_that_leaves_mid_stream_closes_the_upstream_before_its_next_chunk() {
    // The engine sends three chunks, then nothing: only the client's leaving can end the stream.
    let chunk = Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
    let mut mock = Mock::start(Script::Quiet {
        chunks: vec![chunk; 3],
    })
    .await;
    let gw = Gateway::start(mock.addr, "").await;
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
    gw.await_metric(
        "flatshare_gateway_streams_active{class=\"background\"}",
        1.0,
    )
    .await;

    connection.abort();
    drop(body);
    mock.next_drop(Duration::from_secs(2)).await;
    gw.await_metric(&format!("{OUTCOME}\"cancelled\"}}"), 1.0)
        .await;
    gw.await_metric(
        "flatshare_gateway_streams_active{class=\"background\"}",
        0.0,
    )
    .await;
}

#[tokio::test]
async fn a_client_that_leaves_before_the_response_cancels_the_upstream_request() {
    // The engine answers after 10 s: a drop before then is the gateway's cancel.
    let mut mock = Mock::start(Script::Slow(Duration::from_secs(10))).await;
    let gw = Gateway::start(mock.addr, "").await;
    let (mut sender, connection) = connect(&gw.tcp).await;
    let pending = tokio::spawn(sender.send_request(request(
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )));
    let deadline = Instant::now() + Duration::from_secs(2);
    while mock.requests().is_empty() {
        assert!(
            Instant::now() < deadline,
            "the request never reached the mock engine"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    connection.abort();
    pending.abort();
    mock.next_drop(Duration::from_secs(2)).await;
    gw.await_metric(&format!("{OUTCOME}\"cancelled\"}}"), 1.0)
        .await;
    gw.await_metric(
        "flatshare_gateway_requests_total{tenant=\"agents\",route=\"/v1/chat/completions\",class=\"background\",status=\"499\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn a_stalled_reader_is_cut_off_after_the_stall_timeout() {
    let mut mock = Mock::start(Script::Endless {
        chunk: Bytes::from(vec![b'x'; 64 << 10]),
        gap: Duration::ZERO,
    })
    .await;
    let top = "stall_timeout_s = 0.3\nstream_buffer_frames = 4";
    let gw = Gateway::start(mock.addr, top).await;
    let started = Instant::now();
    // The client never reads: its socket fills, then the gateway's buffer, and the stream stalls.
    let (response, _connection) = open(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    let dropped = mock.next_drop(Duration::from_secs(10)).await;
    let after = dropped.saturating_duration_since(started);
    assert!(
        after >= Duration::from_millis(300) && after < Duration::from_secs(5),
        "the upstream was aborted {after:?} after the request"
    );
    gw.await_metric(&format!("{OUTCOME}\"stalled\"}}"), 1.0)
        .await;

    // What was buffered still arrives, and then an error: a cut-off stream must not look complete.
    let mut body = response.into_body();
    let end = loop {
        match body.frame().await {
            Some(Ok(_)) => continue,
            Some(Err(_)) => break "error",
            None => break "clean end",
        }
    };
    assert_eq!(end, "error");
}

#[tokio::test]
async fn an_unreachable_upstream_is_502() {
    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = closed.local_addr().unwrap();
    drop(closed);
    let gw = Gateway::start(addr, "").await;
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
    assert_eq!(error["type"], "server_error");
    assert_eq!(error["code"], "upstream_unavailable");
    assert!(reply.header("x-flatshare-session-id").is_some());
    gw.await_metric(
        "flatshare_gateway_upstream_errors_total{kind=\"connect\"}",
        1.0,
    )
    .await;
    gw.await_metric(&format!("{OUTCOME}\"error\"}}"), 1.0).await;
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 503);
}

#[tokio::test]
async fn no_response_headers_within_the_first_byte_timeout_is_504() {
    let mut mock = Mock::start(Script::Slow(Duration::from_secs(10))).await;
    let gw = Gateway::start(mock.addr, "first_byte_timeout_s = 0.2").await;
    let started = Instant::now();
    let reply = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 504);
    assert_eq!(reply.json()["error"]["code"], "upstream_timeout");
    assert!(started.elapsed() >= Duration::from_millis(200));
    // The engine's request was abandoned too: the gateway closed the connection, which dropped the handler.
    let dropped = mock.next_drop(Duration::from_secs(2)).await;
    assert!(dropped.saturating_duration_since(started) < Duration::from_secs(2));
    gw.await_metric(
        "flatshare_gateway_upstream_errors_total{kind=\"timeout\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn shutdown_stops_accepting_and_drains_in_flight_streams() {
    let chunks: Vec<Bytes> = (0..5)
        .map(|i| Bytes::from(format!("data: {i}\n\n")))
        .collect();
    let expected: Vec<u8> = chunks.concat();
    let mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(100),
    })
    .await;
    let mut gw = Gateway::start(mock.addr, "drain_timeout_s = 5").await;
    let (response, _connection) = open(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    let mut body = response.into_body();
    let mut received = body
        .frame()
        .await
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap()
        .to_vec();

    gw.shut_down();
    let Target::Tcp(addr) = gw.tcp.clone() else {
        unreachable!()
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    while TcpStream::connect(addr).await.is_ok() {
        assert!(
            Instant::now() < deadline,
            "the gateway still accepts after shutdown"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // The stream that was in flight completes.
    while let Some(frame) = body.frame().await {
        received.extend_from_slice(&frame.unwrap().into_data().unwrap());
    }
    assert_eq!(received, expected);
    tokio::time::timeout(Duration::from_secs(2), gw.stopped())
        .await
        .expect("the gateway exits once drained");
}

#[tokio::test]
async fn the_unix_listener_replaces_a_stale_socket_is_private_and_cleans_up() {
    let mock = Mock::start(Script::Json("{}")).await;
    let path = socket_path();
    // A socket file left behind by a process that died.
    drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
    assert!(path.exists());
    let config = Config::from_toml(&gateway_toml(mock.addr, &path, "")).unwrap();
    let mut gw = Gateway::from_config(config.clone(), Arc::new(LogSink)).await;
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);

    // A live socket is never taken over.
    let err = Server::bind(config, Arc::new(LogSink)).await.unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::AddrInUse);

    let reply = call(&gw.unix, "POST", "/v1/chat/completions", &[], CHAT).await;
    assert_eq!(reply.status, 200);
    assert_eq!(mock.last().header("x-vllm-priority"), Some("0"));
    gw.stopped().await;
    assert!(!path.exists(), "the socket file is removed at exit");

    // A path that holds something other than a socket is left alone.
    std::fs::write(&path, "not a socket").unwrap();
    let config = Config::from_toml(&gateway_toml(mock.addr, &path, "")).unwrap();
    assert!(Server::bind(config, Arc::new(LogSink)).await.is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "not a socket");
    std::fs::remove_file(&path).unwrap();
}

/// A response body read to its end: the bytes, and whether it ended with an error rather than cleanly.
async fn read_to_end(target: &Target, headers: &[(&str, &str)]) -> (Vec<u8>, bool) {
    let (response, _connection) = open(target, "POST", "/v1/chat/completions", headers, CHAT).await;
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    loop {
        match body.frame().await {
            Some(Ok(frame)) => bytes.extend_from_slice(&frame.into_data().unwrap()),
            Some(Err(_)) => return (bytes, true),
            None => return (bytes, false),
        }
    }
}

/// The error event the gateway sends after `relayed` when the upstream breaks off an event stream.
fn broken_off(relayed: &[u8], bytes: &[u8]) -> serde_json::Value {
    let event = bytes
        .strip_prefix(relayed)
        .expect("the stream relays the upstream's bytes first");
    let event = std::str::from_utf8(event).unwrap();
    let data = event
        .strip_prefix("data: ")
        .and_then(|data| data.strip_suffix("\n\n"))
        .unwrap_or_else(|| panic!("not one error event: {event:?}"));
    serde_json::from_str(data).unwrap()
}

#[tokio::test]
async fn an_event_stream_the_upstream_breaks_off_ends_with_a_typed_error_then_an_abort() {
    let chunks: Vec<Bytes> = (0..2)
        .map(|i| {
            Bytes::from(format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{i}\"}}}}]}}\n\n"
            ))
        })
        .collect();
    let relayed: Vec<u8> = chunks.concat();
    let mock = Mock::start(Script::Broken {
        content_type: "text/event-stream",
        chunks,
        gap: Duration::from_millis(10),
    })
    .await;
    let gw = Gateway::start(mock.addr, "").await;
    // A realtime stream on the voice plane's socket, as when the engine dies under a call.
    let (bytes, aborted) = read_to_end(&gw.unix, &[]).await;
    assert!(aborted, "a broken stream must not end cleanly");
    let error = broken_off(&relayed, &bytes);
    let message = error["error"]["message"].as_str().unwrap();
    assert!(
        message.starts_with("the upstream stream broke off: "),
        "{message}"
    );
    assert_eq!(
        error,
        serde_json::json!({"error": {"message": message, "type": "upstream_unavailable", "param": null, "code": 502}})
    );
    gw.await_metric(
        "flatshare_gateway_error_events_total{class=\"realtime\",type=\"upstream_unavailable\"}",
        1.0,
    )
    .await;
    gw.await_metric(
        "flatshare_gateway_upstream_errors_total{kind=\"body\"}",
        1.0,
    )
    .await;
    gw.await_metric(
        "flatshare_gateway_request_outcomes_total{class=\"realtime\",outcome=\"error\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn a_break_inside_an_event_ends_that_event_before_the_error() {
    let partial = Bytes::from_static(b"data: {\"choices\":[{\"del");
    let mock = Mock::start(Script::Broken {
        content_type: "text/event-stream",
        chunks: vec![partial.clone()],
        gap: Duration::from_millis(10),
    })
    .await;
    let gw = Gateway::start(mock.addr, "").await;
    let (bytes, aborted) = read_to_end(&gw.tcp, &[AGENTS_AUTH]).await;
    assert!(aborted);
    // The partial event ends at a blank line, so that the error is an event of its own.
    let relayed = [&partial[..], b"\n\n"].concat();
    let error = broken_off(&relayed, &bytes);
    assert_eq!(error["error"]["type"], "upstream_unavailable");
    gw.await_metric(
        "flatshare_gateway_error_events_total{class=\"background\",type=\"upstream_unavailable\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn a_json_body_the_upstream_breaks_off_gets_no_event() {
    let partial = Bytes::from_static(b"{\"id\":\"c-1\",");
    let mock = Mock::start(Script::Broken {
        content_type: "application/json",
        chunks: vec![partial.clone()],
        gap: Duration::from_millis(10),
    })
    .await;
    let gw = Gateway::start(mock.addr, "").await;
    let (bytes, aborted) = read_to_end(&gw.tcp, &[AGENTS_AUTH]).await;
    assert!(aborted);
    assert_eq!(bytes, &partial[..]);
    gw.await_metric(
        "flatshare_gateway_upstream_errors_total{kind=\"body\"}",
        1.0,
    )
    .await;
    let metrics = gw.metrics().await;
    let series =
        "flatshare_gateway_error_events_total{class=\"background\",type=\"upstream_unavailable\"}";
    assert_eq!(metric(&metrics, series), Some(0.0));
}
