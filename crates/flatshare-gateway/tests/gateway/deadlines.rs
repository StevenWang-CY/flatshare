//! Deadlines: each wait a request goes through ends by one, the upstream's work is cancelled when one passes, its
//! place and bytes come back, and it has one outcome. A preempted request's retries keep the deadlines of its receipt.

use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::BodyExt;

use crate::common::*;

/// The outcomes a class's requests had so far, the series that are not zero.
async fn outcomes(gw: &Gateway, class: &str) -> Vec<(String, f64)> {
    let text = gw.metrics().await;
    let prefix = format!("flatshare_gateway_request_outcomes_total{{class=\"{class}\",outcome=\"");
    text.lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .filter_map(|rest| {
            let (outcome, value) = rest.split_once("\"} ")?;
            let value: f64 = value.parse().ok()?;
            (value > 0.0).then(|| (outcome.to_owned(), value))
        })
        .collect()
}

/// Waits until a class's requests have ended with exactly `outcome`, once, and their places are back.
async fn one_outcome(gw: &Gateway, class: &str, outcome: &str) {
    gw.await_metric(
        &format!(
            "flatshare_gateway_request_outcomes_total{{class=\"{class}\",outcome=\"{outcome}\"}}"
        ),
        1.0,
    )
    .await;
    assert_eq!(outcomes(gw, class).await, [(outcome.to_owned(), 1.0)]);
    gw.await_metric(
        &format!("flatshare_gateway_in_flight{{class=\"{class}\"}}"),
        0.0,
    )
    .await;
}

/// A chat completion's response: its status, and its body read to its end, which is an error or a clean end.
async fn stream(target: &Target, headers: &[(&str, &str)]) -> (u16, Vec<u8>, bool, Instant) {
    let (response, _connection) = open(target, "POST", "/v1/chat/completions", headers, CHAT).await;
    let status = response.status().as_u16();
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    loop {
        match body.frame().await {
            Some(Ok(frame)) => bytes.extend_from_slice(&frame.into_data().unwrap()),
            Some(Err(_)) => return (status, bytes, true, Instant::now()),
            None => return (status, bytes, false, Instant::now()),
        }
    }
}

/// The error event that ends a stream after `relayed`.
fn timeout_event(relayed: &[u8], bytes: &[u8]) -> serde_json::Value {
    let event = std::str::from_utf8(
        bytes
            .strip_prefix(relayed)
            .expect("the upstream's bytes first"),
    )
    .unwrap();
    let data = event
        .strip_prefix("data: ")
        .and_then(|data| data.strip_suffix("\n\n"))
        .unwrap_or_else(|| panic!("not one error event: {event:?}"));
    let event: serde_json::Value = serde_json::from_str(data).unwrap();
    assert_eq!(
        (&event["error"]["type"], &event["error"]["code"]),
        (&"upstream_timeout".into(), &504.into()),
        "{event}"
    );
    event
}

fn chunks(n: usize) -> Vec<Bytes> {
    (0..n)
        .map(|i| Bytes::from(format!("data: {{\"n\":{i}}}\n\n")))
        .collect()
}

#[tokio::test]
async fn headers_then_silence_end_at_the_first_content_deadline() {
    let mut mock = Mock::start(Script::Quiet { chunks: Vec::new() }).await;
    let gw = Gateway::start(
        mock.addr,
        "[classes.background]\nfirst_content_timeout_s = 0.3",
    )
    .await;
    let started = Instant::now();
    let (status, bytes, aborted, ended) = stream(&gw.tcp, &[AGENTS_AUTH]).await;
    // The headers came at once; the first content never did.
    assert_eq!(status, 200);
    assert!(aborted, "a stream cut short must not end cleanly");
    let event = timeout_event(b"", &bytes);
    assert!(
        event["error"]["message"]
            .as_str()
            .unwrap()
            .contains("classes.background.first_content_timeout_s (0.3 s)"),
        "{event}"
    );
    let after = ended - started;
    assert!(
        after >= Duration::from_millis(300) && after < Duration::from_secs(2),
        "{after:?}"
    );
    // The engine's request is cancelled: its response was dropped.
    mock.next_drop(Duration::from_secs(2)).await;
    one_outcome(&gw, "background", "timeout_first_content").await;
    let metrics = gw.metrics().await;
    for (series, value) in [
        (
            "flatshare_gateway_upstream_errors_total{kind=\"timeout\"}",
            1.0,
        ),
        (
            "flatshare_gateway_error_events_total{class=\"background\",type=\"upstream_timeout\"}",
            1.0,
        ),
    ] {
        assert_eq!(metric(&metrics, series), Some(value), "{series}");
    }
}

#[tokio::test]
async fn partial_sse_then_silence_end_at_the_idle_deadline() {
    let sent = chunks(2);
    let relayed = sent.concat();
    let mut mock = Mock::start(Script::Quiet { chunks: sent }).await;
    let gw = Gateway::start(mock.addr, "[classes.background]\nidle_timeout_s = 0.3").await;
    let (status, bytes, aborted, _) = stream(&gw.tcp, &[AGENTS_AUTH]).await;
    assert_eq!(status, 200);
    assert!(aborted);
    // Both chunks, then the event: no [DONE], and no clean end.
    let event = timeout_event(&relayed, &bytes);
    assert!(
        event["error"]["message"]
            .as_str()
            .unwrap()
            .contains("idle_timeout_s")
    );
    mock.next_drop(Duration::from_secs(2)).await;
    one_outcome(&gw, "background", "timeout_idle").await;
}

#[tokio::test]
async fn a_stream_that_runs_past_the_total_deadline_is_cut() {
    let mut mock = Mock::start(Script::Sse {
        chunks: chunks(20),
        gap: Duration::from_millis(50),
    })
    .await;
    // The realtime class's: a voice turn's whole reply.
    let gw = Gateway::start(mock.addr, "[classes.realtime]\ntotal_timeout_s = 0.3").await;
    let started = Instant::now();
    let (status, bytes, aborted, ended) = stream(&gw.unix, &[]).await;
    assert_eq!(status, 200);
    assert!(aborted);
    let text = String::from_utf8_lossy(&bytes);
    let relayed = text.matches("\"n\":").count();
    assert!((2..20).contains(&relayed), "{text}");
    let tail = text
        .rsplit_once("\n\ndata: ")
        .map(|(head, _)| format!("{head}\n\n"))
        .unwrap();
    timeout_event(tail.as_bytes(), &bytes);
    let after = ended - started;
    assert!(
        after >= Duration::from_millis(300) && after < Duration::from_secs(2),
        "{after:?}"
    );
    mock.next_drop(Duration::from_secs(2)).await;
    one_outcome(&gw, "realtime", "timeout_total").await;
}

#[tokio::test]
async fn no_response_headers_within_a_classs_headers_deadline_is_504() {
    let mut mock = Mock::start(Script::Slow(Duration::from_secs(10))).await;
    let gw = Gateway::start(mock.addr, "[classes.interactive]\nheaders_timeout_s = 0.3").await;
    let started = Instant::now();
    let reply = call(&gw.tcp, "POST", "/v1/chat/completions", &[BENCH_AUTH], CHAT).await;
    assert_eq!(reply.status, 504);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "upstream_timeout");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("classes.interactive.headers_timeout_s (0.3 s)"),
        "{error}"
    );
    assert!(started.elapsed() >= Duration::from_millis(300));
    mock.next_drop(Duration::from_secs(2)).await;
    one_outcome(&gw, "interactive", "timeout_headers").await;
}

#[tokio::test]
async fn a_slow_upload_gets_408_and_never_goes_upstream() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "[classes.background]\nupload_timeout_s = 0.3").await;
    // Forty bytes announced, one every 100 ms.
    let parts = (0..40)
        .map(|_| (Duration::from_millis(100), b"x".to_vec()))
        .collect();
    let answer = answered(raw(tcp(&gw), head(AGENTS_KEY, "Content-Length: 40"), parts)).await;
    assert_eq!(answer.status(), "408", "{}", answer.text);
    assert_eq!(answer.json()["error"]["code"], "upload_timeout");
    assert!(
        answer.after >= Duration::from_millis(300) && answer.after < Duration::from_secs(2),
        "{:?}",
        answer.after
    );
    assert!(mock.requests().is_empty());
    gw.await_metric(
        "flatshare_gateway_requests_total{tenant=\"agents\",route=\"/v1/chat/completions\",class=\"background\",status=\"408\"}",
        1.0,
    )
    .await;
    gw.await_metric("flatshare_gateway_in_flight{class=\"background\"}", 0.0)
        .await;
    gw.await_metric("flatshare_gateway_buffered_body_bytes", 0.0)
        .await;
}

#[tokio::test]
async fn a_request_not_admitted_by_its_queue_deadline_gets_503_with_retry_after() {
    let mock = Mock::start(Script::Json("{}")).await;
    // The policy's own queue_timeout_s is a minute; the class's deadline comes first.
    let top = "[classes.background]\nqueue_timeout_s = 0.3\n\
               [admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 1";
    let gw = Gateway::start(mock.addr, top).await;
    let first = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (TAG, "bg1"), (GATE, "1")], CHAT);
    mock.await_requests(1).await;
    let started = Instant::now();
    let reply = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH, (TAG, "bg2")],
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 503);
    assert_eq!(reply.json()["error"]["code"], "queue_timeout");
    assert_eq!(reply.header("retry-after"), Some("1"));
    assert!(started.elapsed() >= Duration::from_millis(300));
    gw.await_metric(
        "flatshare_gateway_admission_waiting{class=\"background\"}",
        0.0,
    )
    .await;
    assert_eq!(mock.tags(), ["bg1"], "it never reached the engine");
    mock.release(1);
    assert_eq!(first.await.unwrap().status, 200);
    gw.await_metric("flatshare_gateway_in_flight{class=\"background\"}", 0.0)
        .await;
}

const PREEMPT: &str =
    "[classes.background]\nfirst_content_timeout_s = 1\n[admission]\npolicy = \"preempt-resubmit\"";

#[tokio::test]
async fn a_retry_runs_within_the_deadline_of_the_requests_receipt() {
    let sent = chunks(3);
    let whole = sent.concat();
    let mut mock = Mock::start(Script::Sse {
        chunks: sent,
        gap: Duration::from_millis(5),
    })
    .await;
    let gw = Gateway::start(mock.addr, PREEMPT).await;
    let started = Instant::now();
    let background = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (TAG, "bg"), (GATE, "1")], CHAT);
    mock.await_requests(1).await;
    // A realtime arrival preempts it; its first byte, 100 ms on, lets it go again, and it ends well inside its second.
    let voice = spawn_call(&gw.unix, vec![(TAG, "rt"), (GATE, "1")], CHAT);
    mock.dropped("bg", Duration::from_secs(2)).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    mock.release(1);
    mock.await_requests(3).await;
    mock.release(1);
    let reply = background.await.unwrap();
    assert_eq!(
        (reply.status, &reply.body[..]),
        (http::StatusCode::OK, &whole[..])
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(voice.await.unwrap().status, 200);
    gw.await_metric(
        "flatshare_gateway_admission_resubmits_total{class=\"background\"}",
        1.0,
    )
    .await;
    one_outcome(&gw, "background", "completed").await;
}

#[tokio::test]
async fn a_retry_never_buys_a_request_more_than_its_first_deadline() {
    let mut mock = Mock::start(Script::Sse {
        chunks: chunks(3),
        gap: Duration::from_millis(5),
    })
    .await;
    let gw = Gateway::start(mock.addr, PREEMPT).await;
    let started = Instant::now();
    let background = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (TAG, "bg"), (GATE, "1")], CHAT);
    mock.await_requests(1).await;
    // The realtime request holds its first byte past the background request's second: the background request waits
    // to go again until its first-content deadline, counted from its receipt, not from its preemption.
    let voice = spawn_call(&gw.unix, vec![(TAG, "rt"), (GATE, "1")], CHAT);
    mock.dropped("bg", Duration::from_secs(2)).await;
    let reply = background.await.unwrap();
    let after = started.elapsed();
    assert_eq!(reply.status, 503);
    assert_eq!(reply.json()["error"]["code"], "queue_timeout");
    assert_eq!(reply.header("retry-after"), Some("1"));
    assert!(
        reply.json()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("first_content_timeout_s (1 s)")
    );
    assert!(
        after >= Duration::from_secs(1) && after < Duration::from_millis(1500),
        "{after:?}"
    );
    assert_eq!(mock.tags(), ["bg", "rt"], "never sent again");
    mock.release(1);
    assert_eq!(voice.await.unwrap().status, 200);
    one_outcome(&gw, "background", "timeout_first_content").await;
    gw.await_metric(
        "flatshare_gateway_admission_waiting{class=\"background\"}",
        0.0,
    )
    .await;
}

#[tokio::test]
async fn a_client_that_leaves_while_the_engine_is_slow_to_its_first_token_cancels_it() {
    // The engine holds its first chunk, as while it restores a session's KV from the CPU tier.
    let mut mock = Mock::start(Script::Sse {
        chunks: chunks(3),
        gap: Duration::from_millis(5),
    })
    .await;
    let gw = Gateway::start(mock.addr, "").await;
    let (response, connection) = open(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH, (GATE, "1")],
        CHAT,
    )
    .await;
    assert_eq!(response.status(), 200);
    gw.await_metric("flatshare_gateway_in_flight{class=\"background\"}", 1.0)
        .await;
    connection.abort();
    drop(response);
    mock.next_drop(Duration::from_secs(2)).await;
    one_outcome(&gw, "background", "cancelled").await;
}
