//! The lifecycle policy: lifecycle events through `/v1/events`, then requests. The pool is 1,000 tokens and a turn's
//! headroom 200, so one armed session leaves background requests 800; a background request of about 600 tokens
//! fits alone, and two do not.
//!
//! Its tests: a background request deferred while a session is armed and sent when its reply's first audio is posted,
//! with the realtime request going past it; the critical window's prefill budget, and a lost `playback_started` that
//! `playback_ended` makes up for; aging, oversize and not; a degraded reply's `max_tokens` lowered in the body the
//! engine receives, and to the lower of degrade's cap and the realtime class's `max_reply_tokens` with both; a cough's
//! arming that times out, counted as a false alarm with its token-seconds; a realtime request without events that is
//! its own critical window; a new session in a full table taking the place of one idle session; realtime requests that
//! go while the background queue is full. Breaking the headroom, the throttle, degrade, the oversize count or aging in
//! `flatshare-core` fails at least one of them.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use flatshare_gateway::config::AdmissionConfig;
use flatshare_gateway::{Config, LogSink, Server};

use crate::common::*;

const WAITING: &str = "flatshare_gateway_admission_waiting{class=\"background\"}";
const SESSION: &str = "voice-RM_a1B2c3";
/// The headroom alone.
const HEADROOM: &str = "headroom_tokens = 200\nthrottle = false\ndegrade = false";

/// About 2,400 bytes once the gateway adds its cache salt: 600 tokens at 4 bytes a token, and a cold prefill of
/// 55 ms on the frozen box.
fn six_hundred() -> &'static str {
    format!(r#"{{"model":"qwen","prompt":"{}"}}"#, "x".repeat(2_350)).leak()
}

/// About 900 tokens: more than an armed session leaves background requests, even alone.
fn nine_hundred() -> &'static str {
    format!(r#"{{"model":"qwen","prompt":"{}"}}"#, "x".repeat(3_550)).leak()
}

/// The lifecycle policy on a 1,000-token pool, with its mechanisms' keys.
fn lifecycle(keys: &str) -> String {
    format!("[admission]\npolicy = \"lifecycle\"\npool_tokens = 1000\n{keys}")
}

/// The lifecycle policy's throttle alone, which reads no pool, with a budget of `budget_ms`.
fn throttle_alone(budget_ms: u32) -> String {
    format!(
        "[admission]\npolicy = \"lifecycle\"\nheadroom = false\ndegrade = false\n\
         critical_prefill_budget_ms = {budget_ms}"
    )
}

/// A background request of the agents tenant, named `tag`, whose answer the mock engine holds.
fn background(tag: &'static str) -> Vec<(&'static str, &'static str)> {
    vec![AGENTS_AUTH, (TAG, tag), (GATE, "1")]
}

/// A realtime request of the voice session on the voice socket, named `tag`; held by the engine if `gated`.
fn realtime(tag: &'static str, gated: bool) -> Vec<(&'static str, &'static str)> {
    let mut headers = vec![(TAG, tag), ("x-session-id", SESSION)];
    if gated {
        headers.push((GATE, "1"));
    }
    headers
}

/// Posts lifecycle events of the voice session in one batch, as `(event_id, type)`, as the voice worker does.
async fn post(gw: &Gateway, events: &[(&str, &str)]) {
    let events: Vec<String> = events
        .iter()
        .map(|(id, kind)| {
            format!(
                r#"{{"event_id":"{id}","session_id":"{SESSION}","at_ms":1792650010000,"source":"voice-worker",
                    "type":"{kind}","played_ms":0}}"#
            )
        })
        .collect();
    let body = format!("[{}]", events.join(","));
    let reply = call(&gw.unix, "POST", "/v1/events", &[], &body).await;
    assert_eq!(reply.status, 202, "{:?}", reply.body);
}

fn sessions(state: &str) -> String {
    format!("flatshare_gateway_realtime_sessions{{state=\"{state}\"}}")
}

fn deferred(reason: &str) -> String {
    format!("flatshare_gateway_admission_deferred_total{{reason=\"{reason}\"}}")
}

#[tokio::test]
async fn background_waits_while_a_turn_is_imminent_and_goes_after_it() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, &lifecycle(HEADROOM)).await;
    let running = spawn_call(&gw.tcp, background("bg1"), six_hundred());
    mock.await_requests(1).await;

    // The user starts speaking: the next background turn waits, and the running one goes on.
    post(&gw, &[("vw-1", "user_speech_started")]).await;
    gw.await_metric(&sessions("user_speaking"), 1.0).await;
    let deferred_turn = spawn_call(&gw.tcp, background("bg2"), six_hundred());
    gw.await_metric(WAITING, 1.0).await;
    gw.await_metric(&deferred("headroom"), 1.0).await;
    post(&gw, &[("vw-2", "user_turn_ended")]).await;
    gw.await_metric(&sessions("critical"), 1.0).await;
    // The realtime request goes at once, past the waiting background request.
    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt", false),
        CHAT,
    )
    .await;
    assert_eq!(voice.status, 200);
    assert_eq!(mock.tags(), ["bg1", "rt"]);

    // The reply's first audio ends the turn's claim: the deferred request goes.
    post(&gw, &[("vw-3", "playback_started")]).await;
    mock.await_requests(3).await;
    assert_eq!(mock.tags(), ["bg1", "rt", "bg2"]);
    mock.release(2);
    for reply in [running, deferred_turn] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    post(&gw, &[("vw-4", "playback_ended")]).await;
    gw.await_metric(&sessions("idle"), 1.0).await;
    let metrics = gw.metrics().await;
    for (series, value) in [
        (
            "flatshare_gateway_admission_deferral_seconds_count{reason=\"headroom\"}",
            1.0,
        ),
        ("flatshare_gateway_admission_false_alarms_total", 0.0),
        ("flatshare_gateway_admission_degraded_total", 0.0),
        (
            "flatshare_gateway_admission_aged_total{oversize=\"false\"}",
            0.0,
        ),
        (
            "flatshare_gateway_realtime_state_timeouts_total{state=\"critical\"}",
            0.0,
        ),
    ] {
        assert_eq!(metric(&metrics, series), Some(value), "{series}");
    }
}

#[tokio::test]
async fn the_critical_window_spends_a_budget_of_predicted_prefill_time() {
    let mock = Mock::start(Script::Json("{}")).await;
    // The throttle alone: 100 ms of prefill per window, and each cold 600-token prompt takes 55 ms.
    let gw = Gateway::start(mock.addr, &throttle_alone(100)).await;
    post(&gw, &[("vw-1", "user_turn_ended")]).await;
    gw.await_metric(&sessions("critical"), 1.0).await;
    let first = spawn_call(&gw.tcp, background("bg1"), six_hundred());
    mock.await_requests(1).await;
    let second = spawn_call(&gw.tcp, background("bg2"), six_hundred());
    gw.await_metric(WAITING, 1.0).await;
    gw.await_metric(&deferred("critical"), 1.0).await;
    // A lost playback_started: playback_ended still ends the window.
    post(&gw, &[("vw-2", "playback_ended")]).await;
    mock.await_requests(2).await;
    mock.release(2);
    for reply in [first, second] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    assert_eq!(mock.tags(), ["bg1", "bg2"]);
}

/// A configuration built in code can turn the throttle on without the engine's step budget, which a file cannot
/// (`max_num_batched_tokens` defaults to 512). The gateway refuses it before binding anything, naming the key.
#[tokio::test]
async fn a_throttle_without_a_step_budget_is_refused_by_its_key() {
    let path = socket_path();
    let unused = SocketAddr::from(([127, 0, 0, 1], 9));
    let mut config = Config::from_toml(&gateway_toml(unused, &path, &throttle_alone(100))).unwrap();
    let admission = config.admission.take().unwrap();
    config.admission = Some(AdmissionConfig {
        step_tokens: None,
        ..admission
    });
    let err = Server::bind(config, Arc::new(LogSink)).await.unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(
        err.to_string(),
        "admission: policy lifecycle needs max_num_batched_tokens, unless throttle = false"
    );
    assert!(!path.exists(), "nothing was bound");
}

#[tokio::test]
async fn aging_admits_a_deferred_request_after_max_defer_s() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(
        mock.addr,
        &lifecycle(&format!("{HEADROOM}\nmax_defer_s = 0.3")),
    )
    .await;
    post(&gw, &[("vw-1", "user_speech_started")]).await;
    gw.await_metric(&sessions("user_speaking"), 1.0).await;
    // 900 tokens cannot go beside the turn's headroom even alone: only aging lets them in.
    let started = Instant::now();
    let oversize = spawn_call(&gw.tcp, background("big"), nine_hundred());
    mock.await_requests(1).await;
    assert!(started.elapsed() >= Duration::from_millis(300));
    gw.await_metric(
        "flatshare_gateway_admission_aged_total{oversize=\"true\"}",
        1.0,
    )
    .await;
    // A request that then waits for the headroom ages too, without being oversize: alone it would fit.
    let behind = spawn_call(&gw.tcp, background("small"), six_hundred());
    mock.await_requests(2).await;
    gw.await_metric(
        "flatshare_gateway_admission_aged_total{oversize=\"false\"}",
        1.0,
    )
    .await;
    mock.release(2);
    for reply in [oversize, behind] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    gw.await_metric(
        "flatshare_gateway_admission_deferral_seconds_count{reason=\"headroom\"}",
        2.0,
    )
    .await;
    let refused =
        "flatshare_gateway_admission_refused_total{class=\"background\",reason=\"timeout\"}";
    assert_eq!(metric(&gw.metrics().await, refused), Some(0.0));
}

#[tokio::test]
async fn degrade_caps_a_realtime_reply_when_the_headroom_was_not_kept() {
    let mock = Mock::start(Script::Json("{}")).await;
    // Degrade alone: nothing keeps the background request from taking the pool.
    let keys = "headroom = false\nthrottle = false\nheadroom_tokens = 200\ndegrade_max_tokens = 16";
    let gw = Gateway::start(mock.addr, &lifecycle(keys)).await;
    let large = spawn_call(&gw.tcp, background("bg"), nine_hundred());
    mock.await_requests(1).await;
    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt1", false),
        r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":200}"#,
    )
    .await;
    assert_eq!(voice.status, 200);
    // Chat Completions' both limits, as its engine reads the first that is set.
    let body = mock.last().json();
    assert_eq!(
        (&body["max_tokens"], &body["max_completion_tokens"]),
        (&16.into(), &16.into()),
        "{body}"
    );
    assert_eq!(body["messages"][0]["content"], "hi");
    gw.await_metric("flatshare_gateway_admission_degraded_total", 1.0)
        .await;

    // With the pool free again, a reply is not capped. The request leaves the admission point just after its
    // client has the whole response, and before its outcome is counted.
    mock.release(1);
    assert_eq!(large.await.unwrap().status, 200);
    gw.await_metric(
        "flatshare_gateway_request_outcomes_total{class=\"background\",outcome=\"completed\"}",
        1.0,
    )
    .await;
    call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt2", false),
        CHAT,
    )
    .await;
    let body = mock.last().json();
    assert!(body.get("max_completion_tokens").is_none(), "{body}");
}

#[tokio::test]
async fn a_degraded_reply_takes_the_lower_of_its_caps() {
    let mock = Mock::start(Script::Json("{}")).await;
    let keys = "headroom = false\nthrottle = false\nheadroom_tokens = 200\ndegrade_max_tokens = 16";
    let top = format!(
        "{}\n[classes.realtime]\nmax_reply_tokens = 64",
        lifecycle(keys)
    );
    let gw = Gateway::start(mock.addr, &top).await;
    let asked = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":200}"#;
    // With the pool free, the deployment's cap.
    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt1", false),
        asked,
    )
    .await;
    assert_eq!(voice.status, 200);
    let body = mock.last().json();
    assert_eq!(
        (&body["max_tokens"], &body["max_completion_tokens"]),
        (&64.into(), &64.into())
    );
    // With the pool taken, degrade's, the lower.
    let large = spawn_call(&gw.tcp, background("bg"), nine_hundred());
    mock.await_requests(2).await;
    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt2", false),
        asked,
    )
    .await;
    assert_eq!(voice.status, 200);
    let body = mock.last().json();
    assert_eq!(
        (&body["max_tokens"], &body["max_completion_tokens"]),
        (&16.into(), &16.into())
    );
    gw.await_metric("flatshare_gateway_admission_degraded_total", 1.0)
        .await;
    mock.release(1);
    assert_eq!(large.await.unwrap().status, 200);
}

#[tokio::test]
async fn a_state_no_event_ends_times_out_and_its_arming_is_a_false_alarm() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(
        mock.addr,
        &lifecycle(&format!("{HEADROOM}\nstate_timeout_s = 0.3")),
    )
    .await;
    let running = spawn_call(&gw.tcp, background("bg1"), six_hundred());
    mock.await_requests(1).await;
    // A cough: speech onset, and no turn after it.
    post(&gw, &[("vw-1", "user_speech_started")]).await;
    let held = spawn_call(&gw.tcp, background("bg2"), six_hundred());
    gw.await_metric(WAITING, 1.0).await;
    gw.await_metric(
        "flatshare_gateway_realtime_state_timeouts_total{state=\"user_speaking\"}",
        1.0,
    )
    .await;
    mock.await_requests(2).await;
    gw.await_metric(&sessions("user_speaking"), 0.0).await;
    let metrics = gw.metrics().await;
    assert_eq!(
        metric(&metrics, "flatshare_gateway_admission_false_alarms_total"),
        Some(1.0)
    );
    // About 600 tokens held back for most of the 0.3 s.
    let cost = metric(
        &metrics,
        "flatshare_gateway_admission_false_alarm_token_seconds_total",
    )
    .unwrap();
    assert!((60.0..=600.0).contains(&cost), "{cost}");
    mock.release(2);
    for reply in [running, held] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
}

#[tokio::test]
async fn without_events_a_realtime_request_is_its_own_critical_window() {
    let mock = Mock::start(Script::Json("{}")).await;
    // The throttle admits no background prefill in a window.
    let gw = Gateway::start(mock.addr, &throttle_alone(0)).await;
    let voice = spawn_call(&gw.unix, realtime("rt", true), CHAT);
    mock.await_requests(1).await;
    gw.await_metric(&sessions("critical"), 1.0).await;
    let deferred_turn = spawn_call(&gw.tcp, background("bg"), six_hundred());
    gw.await_metric(WAITING, 1.0).await;
    // The reply ends, and with it the window: nothing else would end it.
    mock.release(1);
    assert_eq!(voice.await.unwrap().status, 200);
    mock.await_requests(2).await;
    gw.await_metric(&sessions("critical"), 0.0).await;
    gw.await_metric(&sessions("idle"), 0.0).await;
    mock.release(1);
    assert_eq!(deferred_turn.await.unwrap().status, 200);
}

#[tokio::test]
async fn a_new_session_in_a_full_table_takes_the_place_of_one_idle_session() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, &lifecycle(HEADROOM)).await;
    post(&gw, &[("vw-1", "user_speech_started")]).await;
    // 4,095 more sessions, idle, fill the table.
    let idle: Vec<String> = (1..4096)
        .map(|n| {
            format!(
                r#"{{"event_id":"idle-{n}","session_id":"voice-{n}","at_ms":1792650010000,"source":"voice-worker",
                    "type":"playback_ended","played_ms":0}}"#
            )
        })
        .collect();
    let batch = format!("[{}]", idle.join(","));
    let reply = call(&gw.unix, "POST", "/v1/events", &[], &batch).await;
    assert_eq!(reply.status, 202, "{:?}", reply.body);
    gw.await_metric(&sessions("idle"), 4095.0).await;
    // A new session's turn takes the place of one idle session, not of all of them.
    let new = r#"{"event_id":"new-1","session_id":"voice-new","at_ms":1792650010000,"source":"voice-worker",
                  "type":"user_turn_ended"}"#;
    let reply = call(&gw.unix, "POST", "/v1/events", &[], new).await;
    assert_eq!(reply.status, 202, "{:?}", reply.body);
    let metrics = gw.metrics().await;
    let count = |state: &str| metric(&metrics, &sessions(state));
    assert_eq!(
        [count("idle"), count("user_speaking"), count("critical")],
        [Some(4094.0), Some(1.0), Some(1.0)]
    );
}

#[tokio::test]
async fn realtime_is_never_queued_or_refused() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(
        mock.addr,
        &lifecycle(&format!("{HEADROOM}\nqueue_limit = 1")),
    )
    .await;
    post(
        &gw,
        &[("vw-1", "user_speech_started"), ("vw-2", "user_turn_ended")],
    )
    .await;
    gw.await_metric(&sessions("critical"), 1.0).await;
    let oversize = spawn_call(&gw.tcp, background("big"), nine_hundred());
    gw.await_metric(WAITING, 1.0).await;
    let full = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &background("bg2"),
        CHAT,
    )
    .await;
    assert_eq!(full.status, 429);
    for tag in ["rt1", "rt2"] {
        let voice = call(
            &gw.unix,
            "POST",
            "/v1/chat/completions",
            &realtime(tag, false),
            CHAT,
        )
        .await;
        assert_eq!(voice.status, 200);
    }
    assert_eq!(mock.tags(), ["rt1", "rt2"]);
    post(&gw, &[("vw-3", "session_closed")]).await;
    mock.await_requests(3).await;
    mock.release(1);
    assert_eq!(oversize.await.unwrap().status, 200);
    let refused =
        "flatshare_gateway_admission_refused_total{class=\"realtime\",reason=\"queue_full\"}";
    assert_eq!(metric(&gw.metrics().await, refused), Some(0.0));
}
