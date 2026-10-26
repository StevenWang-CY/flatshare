//! A drain: what a stopping or reloading gateway does with the requests in flight. Realtime and interactive ones
//! finish within the drain timeout; background ones are cut `background_drain_s` into it, so that no agent turn
//! holds up a stop or a reload.

use std::time::{Duration, Instant};

use bytes::Bytes;

use crate::common::*;

const DRAIN: &str = "drain_timeout_s = 5\nbackground_drain_s = 0.3";
const GRACE: Duration = Duration::from_millis(300);
const WAITING: &str = "flatshare_gateway_admission_waiting{class=\"background\"}";

fn outcome(class: &str, outcome: &str) -> String {
    format!("flatshare_gateway_request_outcomes_total{{class=\"{class}\",outcome=\"{outcome}\"}}")
}

#[tokio::test]
async fn a_drain_cuts_background_streams_and_lets_realtime_and_interactive_ones_finish() {
    let chunks: Vec<Bytes> = (0..30)
        .map(|i| {
            Bytes::from(format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{i}\"}}}}]}}\n\n"
            ))
        })
        .collect();
    let whole: Vec<u8> = chunks.concat();
    let mut mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(50),
    })
    .await;
    let mut gw = Gateway::start(mock.addr, DRAIN).await;
    let realtime = begin(&gw.unix, &[(TAG, "rt")]).await;
    let interactive = begin(&gw.tcp, &[BENCH_AUTH, (TAG, "in")]).await;
    let background = begin(&gw.tcp, &[AGENTS_AUTH, (TAG, "bg")]).await;

    let stopping = Instant::now();
    gw.shut_down();
    let (realtime, interactive, background) =
        (finish(realtime), finish(interactive), finish(background));

    // The background stream ends at the cut: the bytes relayed so far, then the error event, then an abort.
    let (bytes, aborted, ended) = soon(background).await.unwrap();
    assert!(aborted, "a cut stream must not end cleanly");
    let cut_after = ended - stopping;
    assert!(
        cut_after >= GRACE && cut_after < GRACE * 3,
        "the background stream ended {cut_after:?} into the drain"
    );
    let needle = b"data: {\"error\":";
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("an error event");
    assert!(
        whole.starts_with(&bytes[..at]) && bytes[..at].ends_with(b"\n\n"),
        "whole events first"
    );
    let event = std::str::from_utf8(&bytes[at..]).unwrap();
    let data = event
        .strip_prefix("data: ")
        .and_then(|data| data.strip_suffix("\n\n"))
        .unwrap();
    let error: serde_json::Value = serde_json::from_str(data).unwrap();
    let message = error["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("background requests end 0.3s into its drain"),
        "{message}"
    );
    assert_eq!(
        error,
        serde_json::json!({"error": {"message": message, "type": "draining", "param": null, "code": 503}})
    );
    // Its engine request was dropped at the cut.
    let dropped = mock.dropped("bg", Duration::from_secs(2)).await;
    assert!(dropped - stopping < GRACE * 3);

    // The realtime and the interactive stream finish whole.
    for stream in [realtime, interactive] {
        let (bytes, aborted, _) = stream.await.unwrap();
        assert!(!aborted);
        assert_eq!(bytes, whole);
    }
    tokio::time::timeout(Duration::from_secs(2), gw.stopped())
        .await
        .expect("the gateway exits once drained");
    let metrics = gw.exposition.encode();
    for (series, value) in [
        (outcome("background", "drained"), 1.0),
        (outcome("realtime", "completed"), 1.0),
        (outcome("interactive", "completed"), 1.0),
        (
            "flatshare_gateway_error_events_total{class=\"background\",type=\"draining\"}".into(),
            1.0,
        ),
        (
            "flatshare_gateway_error_events_total{class=\"background\",type=\"upstream_unavailable\"}"
                .into(),
            0.0,
        ),
    ] {
        assert_eq!(metric(&metrics, &series), Some(value), "{series}");
    }
}

#[tokio::test]
async fn a_background_request_without_a_response_at_the_cut_gets_503_with_retry_after() {
    let mut mock = Mock::start(Script::Slow(Duration::from_secs(10))).await;
    let mut gw = Gateway::start(mock.addr, DRAIN).await;
    let reply = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (TAG, "bg")], CHAT);
    mock.await_requests(1).await;
    let stopping = Instant::now();
    gw.shut_down();
    let reply = soon(reply).await.unwrap();
    let answered = stopping.elapsed();
    assert!(
        answered >= GRACE && answered < GRACE * 3,
        "answered {answered:?} into the drain"
    );
    assert_eq!(reply.status, 503);
    assert_eq!(reply.header("retry-after"), Some("1"));
    let error = &reply.json()["error"];
    assert_eq!(
        (&error["type"], &error["code"]),
        (&"server_error".into(), &"draining".into())
    );
    // The engine's request was abandoned, and the drain ends with it.
    mock.dropped("bg", Duration::from_secs(2)).await;
    tokio::time::timeout(Duration::from_secs(2), gw.stopped())
        .await
        .expect("the gateway exits once drained");
    let metrics = gw.exposition.encode();
    assert_eq!(
        metric(&metrics, &outcome("background", "drained")),
        Some(1.0)
    );
}

#[tokio::test]
async fn background_requests_still_queued_at_the_cut_get_503_and_never_go_upstream() {
    let mock = Mock::start(Script::Json("{}")).await;
    let top =
        format!("{DRAIN}\n[admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 1");
    let mut gw = Gateway::start(mock.addr, &top).await;
    let held = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (TAG, "bg1"), (GATE, "1")], CHAT);
    mock.await_requests(1).await;
    let queued = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (TAG, "bg2")], CHAT);
    gw.await_metric(WAITING, 1.0).await;
    gw.shut_down();
    for reply in [held, queued] {
        let reply = soon(reply).await.unwrap();
        assert_eq!(reply.status, 503);
        assert_eq!(reply.header("retry-after"), Some("1"));
        assert_eq!(reply.json()["error"]["code"], "draining");
    }
    tokio::time::timeout(Duration::from_secs(2), gw.stopped())
        .await
        .expect("the gateway exits once drained");
    assert_eq!(
        mock.tags(),
        ["bg1"],
        "the waiting request never reached the engine"
    );
    let metrics = gw.exposition.encode();
    // Only the request that went upstream has an outcome; both were answered 503.
    assert_eq!(
        metric(&metrics, &outcome("background", "drained")),
        Some(1.0)
    );
    let answered = "flatshare_gateway_requests_total{tenant=\"agents\",route=\"/v1/chat/completions\",\
                    class=\"background\",status=\"503\"}";
    assert_eq!(metric(&metrics, answered), Some(2.0));
    assert_eq!(metric(&metrics, WAITING), Some(0.0));
}

#[tokio::test]
async fn a_background_request_that_ends_within_its_grace_is_not_cut() {
    let chunks: Vec<Bytes> = (0..3)
        .map(|i| Bytes::from(format!("data: {i}\n\n")))
        .collect();
    let whole: Vec<u8> = chunks.concat();
    let mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(50),
    })
    .await;
    let mut gw = Gateway::start(mock.addr, DRAIN).await;
    let background = begin(&gw.tcp, &[AGENTS_AUTH]).await;
    gw.shut_down();
    let (bytes, aborted, _) = finish(background).await.unwrap();
    assert!(!aborted);
    assert_eq!(bytes, whole);
    gw.stopped().await;
    let metrics = gw.exposition.encode();
    assert_eq!(
        metric(&metrics, &outcome("background", "completed")),
        Some(1.0)
    );
    assert_eq!(
        metric(&metrics, &outcome("background", "drained")),
        Some(0.0)
    );
}
