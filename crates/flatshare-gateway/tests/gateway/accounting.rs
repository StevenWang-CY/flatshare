//! Failure-inclusive accounting: every inference request offered once, admitted once if it went upstream and ended
//! once; the service level's valid and good requests, which failures, timeouts and hangs lower; and the engine's state,
//! which the inference canary and the requests' successes drive and `GET /ready` answers by.

use std::time::Duration;

use crate::common::*;

const VALID: &str = "flatshare_gateway_sli_valid_total{class=\"realtime\"}";
const GOOD: &str = "flatshare_gateway_sli_good_total{class=\"realtime\"}";

/// A realtime request on the voice socket whose client leaves `after` its sending, before any response: the engine
/// holds its answer.
async fn left_after(gw: &Gateway, mock: &mut Mock, after: Duration) {
    let (mut sender, connection) = connect(&gw.unix).await;
    let pending = tokio::spawn(sender.send_request(request(
        "POST",
        "/v1/chat/completions",
        &[(GATE, "1"), (TAG, "leaves")],
        CHAT,
    )));
    tokio::time::sleep(after).await;
    connection.abort();
    pending.abort();
    mock.dropped("leaves", Duration::from_secs(2)).await;
}

/// A realtime request on the voice socket.
async fn voice(gw: &Gateway, headers: &[(&str, &str)]) -> Reply {
    call(&gw.unix, "POST", "/v1/chat/completions", headers, CHAT).await
}

fn terminal(outcome: &str) -> String {
    format!(
        "flatshare_gateway_requests_terminal_total{{tenant=\"voice\",route=\"/v1/chat/completions\",\
         class=\"realtime\",outcome=\"{outcome}\"}}"
    )
}

#[tokio::test]
async fn failures_timeouts_and_hangs_lower_the_service_level() {
    let mut mock = Mock::start(Script::Json("{}")).await;
    // A hang ends at the realtime first-content deadline; the objective is the default 250 ms.
    let gw = Gateway::start(
        mock.addr,
        "[classes.realtime]\nfirst_content_timeout_s = 0.6",
    )
    .await;
    // Good: answered at once.
    assert_eq!(voice(&gw, &[]).await.status, 200);
    // Bad: the engine fails it.
    assert_eq!(voice(&gw, &[(STATUS, "500")]).await.status, 500);
    // Bad: completed, but its first content 400 ms after its receipt, past the 250 ms objective.
    let slow = spawn_call(&gw.unix, vec![(GATE, "1"), (TAG, "slow")], CHAT);
    mock.await_requests(3).await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    mock.release(1);
    assert_eq!(slow.await.unwrap().status, 200);
    // Bad: a hang, which its deadline ends.
    let hang = voice(&gw, &[(GATE, "1"), (TAG, "hang")]).await;
    assert_eq!(hang.status, 504);
    mock.dropped("hang", Duration::from_secs(2)).await;
    // Bad: its client left 400 ms in, past the objective, with nothing yet.
    left_after(&gw, &mut mock, Duration::from_millis(400)).await;
    // Not valid: the client's faults, and a client that left before the objective passed.
    assert_eq!(voice(&gw, &[(STATUS, "400")]).await.status, 400);
    let invalid = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &[],
        r#"{"messages":[],"n":8}"#,
    )
    .await;
    assert_eq!(invalid.status, 400);
    left_after(&gw, &mut mock, Duration::from_millis(50)).await;

    gw.await_metric(VALID, 5.0).await;
    let metrics = gw.metrics().await;
    assert_eq!(metric(&metrics, GOOD), Some(1.0), "{metrics}");
    for (outcome, n) in [
        ("completed", 2.0),
        ("upstream_server_error", 1.0),
        ("timeout_first_content", 1.0),
        ("cancelled", 2.0),
        ("upstream_client_error", 1.0),
        ("invalid", 1.0),
    ] {
        assert_eq!(metric(&metrics, &terminal(outcome)), Some(n), "{outcome}");
    }
    // Every request offered once and ended once.
    let offered = "flatshare_gateway_requests_offered_total{tenant=\"voice\",route=\"/v1/chat/completions\",\
                   class=\"realtime\"}";
    assert_eq!(metric(&metrics, offered), Some(8.0));
    // The other classes have no objective: their completed requests are good.
    let agents = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    assert_eq!(agents.status, 200);
    gw.await_metric(
        "flatshare_gateway_sli_good_total{class=\"background\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn every_request_is_offered_once_admitted_once_and_ends_once() {
    let mock = Mock::start(Script::Json("{}")).await;
    let top =
        "max_in_flight = 2\n[admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 4";
    let gw = Gateway::start(mock.addr, top).await;
    // Unauthenticated, and an unknown class: offered and ended with what is known of them.
    assert_eq!(
        call(&gw.tcp, "POST", "/v1/completions", &[], CHAT)
            .await
            .status,
        401
    );
    let unknown = [AGENTS_AUTH, ("x-flatshare-class", "urgent")];
    assert_eq!(
        call(&gw.tcp, "POST", "/v1/responses", &unknown, CHAT)
            .await
            .status,
        400
    );
    // One in flight under the policy, one realtime past it, and one more refused at the intake.
    let held = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (GATE, "1")], CHAT);
    mock.await_requests(1).await;
    let in_flight = |text: &str| {
        let offered: f64 = text
            .lines()
            .filter(|line| line.starts_with("flatshare_gateway_requests_offered_total{"))
            .map(|line| line.rsplit(' ').next().unwrap().parse::<f64>().unwrap())
            .sum();
        let ended: f64 = text
            .lines()
            .filter(|line| line.starts_with("flatshare_gateway_requests_terminal_total{"))
            .map(|line| line.rsplit(' ').next().unwrap().parse::<f64>().unwrap())
            .sum();
        offered - ended
    };
    assert_eq!(
        in_flight(&gw.metrics().await),
        1.0,
        "the request held at the engine"
    );
    let voice = spawn_call(&gw.unix, vec![(GATE, "1")], CHAT);
    mock.await_requests(2).await;
    let refused = call(&gw.tcp, "POST", "/v1/chat/completions", &[BENCH_AUTH], CHAT).await;
    assert_eq!(refused.status, 503);
    mock.release(2);
    assert_eq!(held.await.unwrap().status, 200);
    assert_eq!(voice.await.unwrap().status, 200);
    gw.await_metric("flatshare_gateway_in_flight{class=\"background\"}", 0.0)
        .await;
    gw.await_metric("flatshare_gateway_in_flight{class=\"realtime\"}", 0.0)
        .await;
    let text = gw.metrics().await;
    assert_eq!(in_flight(&text), 0.0, "{text}");
    for (series, value) in [
        (
            "flatshare_gateway_requests_offered_total{tenant=\"\",route=\"/v1/completions\",class=\"\"}",
            1.0,
        ),
        (
            "flatshare_gateway_requests_terminal_total{tenant=\"\",route=\"/v1/completions\",class=\"\",outcome=\"invalid\"}",
            1.0,
        ),
        (
            "flatshare_gateway_requests_terminal_total{tenant=\"agents\",route=\"/v1/responses\",class=\"\",outcome=\"invalid\"}",
            1.0,
        ),
        // Admitted through the policy, which holds background, and past it, which does not hold realtime.
        (
            "flatshare_gateway_requests_admitted_total{tenant=\"agents\",route=\"/v1/chat/completions\",class=\"background\",admission=\"policy\"}",
            1.0,
        ),
        (
            "flatshare_gateway_requests_admitted_total{tenant=\"voice\",route=\"/v1/chat/completions\",class=\"realtime\",admission=\"none\"}",
            1.0,
        ),
        (
            "flatshare_gateway_requests_terminal_total{tenant=\"bench\",route=\"/v1/chat/completions\",class=\"interactive\",outcome=\"refused_in_flight\"}",
            1.0,
        ),
        (
            "flatshare_gateway_request_outcomes_total{class=\"interactive\",outcome=\"refused_in_flight\"}",
            1.0,
        ),
    ] {
        assert_eq!(metric(&text, series), Some(value), "{series}");
    }
    // A refusal for the gateway's capacity is the service's failure; the tenant's own bound would be the client's.
    assert_eq!(
        metric(
            &text,
            "flatshare_gateway_sli_valid_total{class=\"interactive\"}"
        ),
        Some(1.0)
    );
}

const STATE: &str = "flatshare_gateway_engine_state{state=";

async fn state_is(gw: &Gateway, state: &str) {
    gw.await_metric(&format!("{STATE}\"{state}\"}}"), 1.0).await;
}

async fn ready(gw: &Gateway) -> (u16, String) {
    let reply = call(&gw.tcp, "GET", "/ready", &[], "").await;
    (
        reply.status.as_u16(),
        String::from_utf8(reply.body.to_vec()).unwrap(),
    )
}

#[tokio::test]
async fn the_canary_drives_the_engine_states_against_a_failing_then_recovering_engine() {
    let mock = Mock::start(Script::Json(r#"{"choices":[{"text":"o"}]}"#)).await;
    mock.set_failing(true);
    let top = "[canary]\ninterval_s = 0.1\ntimeout_s = 1\ndegraded_after = 2";
    let mut gw = Gateway::start(mock.addr, top).await;
    // Warming: its /health answers, but no inference has succeeded.
    state_is(&gw, "warming").await;
    gw.await_metric(
        "flatshare_gateway_canary_probes_total{result=\"failure\"}",
        2.0,
    )
    .await;
    assert_eq!(ready(&gw).await, (503, "engine warming\n".into()));
    // The engine recovers: the canary's first success serves.
    mock.set_failing(false);
    state_is(&gw, "serving").await;
    assert_eq!(ready(&gw).await, (200, "ready\n".into()));
    // It fails again: two canaries in a row, with nothing succeeding, degrade it.
    mock.set_failing(true);
    state_is(&gw, "degraded").await;
    assert_eq!(ready(&gw).await, (503, "engine degraded\n".into()));
    mock.set_failing(false);
    state_is(&gw, "serving").await;
    // The canary's requests are one token at the lowest class's priority, and no client's.
    let canary = mock
        .requests()
        .into_iter()
        .find(|request| request.header(CANARY).is_some())
        .unwrap();
    assert_eq!(canary.path, "/v1/completions");
    assert_eq!(canary.header("x-vllm-priority"), Some("2"));
    assert_eq!(canary.json()["max_tokens"], 1);
    let metrics = gw.metrics().await;
    assert!(
        !metrics
            .lines()
            .any(|line| line.starts_with("flatshare_gateway_requests_offered_total")),
        "a canary is not a request"
    );
    // Drained once the gateway drains.
    gw.shut_down();
    let drained = format!("{STATE}\"drained\"}}");
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while metric(&gw.exposition.encode(), &drained) != Some(1.0) {
        assert!(std::time::Instant::now() < deadline, "not drained");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    gw.stopped().await;
}

#[tokio::test]
async fn a_clients_success_is_a_passing_canary_and_the_canary_waits_while_requests_succeed() {
    let mock = Mock::start(Script::Json("{}")).await;
    // Only the canary fails: the clients' requests are what show the engine serving.
    mock.set_failing_canary(true);
    let top = "[canary]\ninterval_s = 0.3\ntimeout_s = 1\ndegraded_after = 1";
    let gw = Gateway::start(mock.addr, top).await;
    state_is(&gw, "warming").await;
    let voice = || call(&gw.unix, "POST", "/v1/chat/completions", &[], CHAT);
    assert_eq!(voice().await.status, 200);
    state_is(&gw, "serving").await;
    // While requests succeed, more often than the interval, no canary is sent.
    let before = mock.canaries();
    for _ in 0..10 {
        assert_eq!(voice().await.status, 200);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(mock.canaries(), before, "a canary while requests succeeded");
    // Idle, the canary asks after its interval, fails, and with degraded_after = 1 degrades the engine; the next
    // client's success serves it again.
    state_is(&gw, "degraded").await;
    assert!(mock.canaries() > before);
    assert_eq!(voice().await.status, 200);
    state_is(&gw, "serving").await;
}
