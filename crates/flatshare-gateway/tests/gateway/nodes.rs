//! Several GPU nodes behind one gateway (`[[nodes]]`), on mock engines and mock speech servers: sticky placement, call
//! slots, agents' bounds and calls that fill the nodes in order, a node failing as a whole pipeline (its engine, or
//! only its TTS), sessions moving by their next request, the 503 when the survivors have no call slot, agents deferred
//! on a node that covers for a failed one, a realtime request sent once more before its first content and never after,
//! `/nodes`, `POST /v1/sessions`, and a gateway without `[[nodes]]` unchanged.

use std::time::Duration;

use bytes::Bytes;
use http_body_util::BodyExt;

use crate::common::*;

fn healthy(node: &str) -> String {
    format!("flatshare_gateway_node_healthy{{node=\"{node}\"}}")
}

fn covering(node: &str) -> String {
    format!("flatshare_gateway_node_covering{{node=\"{node}\"}}")
}

fn placements(node: &str, class: &str, reason: &str) -> String {
    format!(
        "flatshare_gateway_placements_total{{node=\"{node}\",class=\"{class}\",reason=\"{reason}\"}}"
    )
}

async fn all_healthy(gw: &Gateway, nodes: &[&TestNode]) {
    for node in nodes {
        gw.await_metric(&healthy(node.name), 1.0).await;
    }
}

/// `POST /v1/sessions` for a call of the voice tenant (trusted, on the socket).
async fn place(gw: &Gateway, session: &str) -> Reply {
    let body = format!(r#"{{"session_id":"{session}","class":"realtime"}}"#);
    call(&gw.unix, "POST", "/v1/sessions", &[], &body).await
}

/// The node `POST /v1/sessions` answers for `session`.
async fn node_of(gw: &Gateway, session: &str) -> String {
    let reply = place(gw, session).await;
    assert_eq!(reply.status, 200, "{:?}", reply.body);
    reply.json()["node"].as_str().unwrap().to_owned()
}

/// A voice turn of the call `session`.
async fn turn(gw: &Gateway, session: &str) -> Reply {
    let headers = [("x-session-id", session)];
    call(&gw.unix, "POST", "/v1/chat/completions", &headers, CHAT).await
}

/// A request of the agent session `session`, of the agents tenant (trusted, background).
async fn agent(gw: &Gateway, session: &str) -> Reply {
    let headers = [AGENTS_AUTH, ("x-session-id", session)];
    call(&gw.tcp, "POST", "/v1/chat/completions", &headers, CHAT).await
}

/// The voice worker's `session_closed` for `session`.
async fn close(gw: &Gateway, session: &str) {
    let body = format!(
        r#"{{"event_id":"close-{session}","session_id":"{session}","at_ms":1792650010000,
            "source":"voice-worker","type":"session_closed"}}"#
    );
    let reply = call(&gw.unix, "POST", "/v1/events", &[], &body).await;
    assert_eq!(reply.status, 202, "{:?}", reply.body);
}

/// The agents tenant's `session_closed` for its session `session`.
async fn agent_closed(gw: &Gateway, session: &str) {
    let body = format!(
        r#"{{"event_id":"close-{session}","session_id":"{session}","at_ms":1792650010000,
            "source":"client","type":"session_closed"}}"#
    );
    let reply = call(&gw.tcp, "POST", "/v1/events", &[AGENTS_AUTH], &body).await;
    assert_eq!(reply.status, 202, "{:?}", reply.body);
}

async fn nodes_view(gw: &Gateway) -> serde_json::Value {
    let reply = call(&gw.unix, "GET", "/nodes", &[], "").await;
    assert_eq!(reply.status, 200, "{:?}", reply.body);
    reply.json()
}

/// A 503 that names `code` and tells its client to come back in a second.
fn unplaced(reply: &Reply, code: &str) {
    assert_eq!(reply.status, 503, "{:?}", reply.body);
    assert_eq!(reply.json()["error"]["code"], code, "{:?}", reply.body);
    assert_eq!(reply.header("retry-after"), Some("1"));
}

#[tokio::test]
async fn without_nodes_the_gateway_places_nothing_and_exports_its_one_engine() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    let session = r#"{"session_id":"call-1","class":"realtime"}"#;
    for (method, path) in [("POST", "/v1/sessions"), ("GET", "/nodes")] {
        let reply = call(&gw.unix, method, path, &[], session).await;
        assert_eq!(reply.status, 404, "{method} {path}");
        assert_eq!(reply.json()["error"]["code"], "not_found");
    }
    let metrics = gw.metrics().await;
    assert!(
        metrics.contains("\nflatshare_gateway_node_ready 1\n"),
        "{metrics}"
    );
    for family in ["node_healthy", "placements", "node_sessions"] {
        assert!(
            !metrics.contains(&format!("flatshare_gateway_{family}")),
            "{family}"
        );
    }
    assert_eq!(turn(&gw, "call-1").await.status, 200);
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
async fn sessions_keep_their_node_and_calls_take_its_call_slots() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    let gw = Gateway::start_nodes(&[&a, &b], "max_calls = 1", "", "").await;
    all_healthy(&gw, &[&a, &b]).await;
    // A call is placed before it speaks, on the node with the fewest calls, and learns its speech servers.
    let first = place(&gw, "call-1").await;
    assert_eq!(first.status, 200, "{:?}", first.body);
    let placed = first.json();
    assert_eq!(
        (
            &placed["node"],
            &placed["class"],
            &placed["stt"],
            &placed["tts"]
        ),
        (
            &"a".into(),
            &"realtime".into(),
            &a.stt.upstream.clone().into(),
            &a.tts.upstream.clone().into()
        )
    );
    assert_eq!(
        node_of(&gw, "call-1").await,
        "a",
        "asked again, the same node"
    );
    assert_eq!(node_of(&gw, "call-2").await, "b");
    // Every node has its max_calls: a third call does not fit.
    unplaced(&place(&gw, "call-3").await, "no_call_slot");
    gw.await_metric(
        "flatshare_gateway_placements_refused_total{class=\"realtime\",reason=\"no_call_slot\"}",
        1.0,
    )
    .await;
    // Each call's turns go to its node, and only there.
    for _ in 0..3 {
        assert_eq!(turn(&gw, "call-1").await.status, 200);
    }
    assert_eq!(turn(&gw, "call-2").await.status, 200);
    assert_eq!((a.requests(), b.requests()), (3, 1));
    // Agents' sessions spread over the nodes, the fewest agents' sessions first, and each stays where it went.
    for session in ["agent-1", "agent-2", "agent-1", "agent-1"] {
        assert_eq!(agent(&gw, session).await.status, 200);
    }
    assert_eq!((a.requests(), b.requests()), (6, 2));
    let view = nodes_view(&gw).await;
    let first = &view["nodes"][0];
    assert_eq!(first["name"], "a");
    assert_eq!(first["verdict"], "healthy");
    assert_eq!(
        first["sessions"],
        serde_json::json!({"realtime": 1, "interactive": 0, "background": 1})
    );
    assert_eq!(first["tts"]["check"], true);
    assert_eq!(view["sessions"], 4);
    // The call's end frees its slot.
    close(&gw, "call-1").await;
    assert_eq!(node_of(&gw, "call-3").await, "a");
    gw.await_metric(&placements("a", "realtime", "new"), 2.0)
        .await;
}

#[tokio::test]
async fn a_node_whose_tts_fails_takes_no_new_call_and_its_calls_move_by_their_next_request() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    // While b covers for a, one agent's request at a time goes on b.
    let placement = "[placement]\nhealth_interval_s = 0.02\nunhealthy_after = 2\nhealthy_after = 2\n\
                     failover_background_max_in_flight = 1\n";
    let gw = Gateway::start_nodes(&[&a, &b], "", "", placement).await;
    all_healthy(&gw, &[&a, &b]).await;
    assert_eq!(node_of(&gw, "call-1").await, "a");
    assert_eq!(turn(&gw, "call-1").await.status, 200);
    // Only its TTS fails: the node is unhealthy, its engine serving.
    a.tts.set_healthy(false);
    gw.await_metric(&healthy("a"), 0.0).await;
    let view = nodes_view(&gw).await;
    let node = &view["nodes"][0];
    assert_eq!(
        (
            &node["verdict"],
            &node["tts"]["check"],
            &node["engine"]["check"]
        ),
        (&"unhealthy".into(), &false.into(), &true.into())
    );
    gw.await_metric(
        "flatshare_gateway_node_component_healthy{node=\"a\",component=\"tts\"}",
        0.0,
    )
    .await;
    // The call's next turn moves it, counted, and its next POST /v1/sessions names the new node.
    assert_eq!(turn(&gw, "call-1").await.status, 200);
    assert_eq!((a.requests(), b.requests()), (1, 1));
    gw.await_metric(&placements("b", "realtime", "moved"), 1.0)
        .await;
    assert_eq!(node_of(&gw, "call-1").await, "b");
    // New calls and agents go to the healthy node, which covers for a: its agents one at a time.
    assert_eq!(node_of(&gw, "call-2").await, "b");
    gw.await_metric(&covering("b"), 1.0).await;
    assert_eq!(agent(&gw, "agent-1").await.status, 200);
    assert_eq!(b.requests(), 2);
    // Healthy again after healthy_after passed checks; the moved call stays where it is.
    a.tts.set_healthy(true);
    gw.await_metric(&healthy("a"), 1.0).await;
    assert_eq!(node_of(&gw, "call-1").await, "b");
    assert_eq!(node_of(&gw, "call-3").await, "a");
}

#[tokio::test]
async fn a_node_whose_engine_fails_its_canary_is_unhealthy_and_its_requests_go_elsewhere() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    let canary = "[canary]\ninterval_s = 0.05\ntimeout_s = 1\ndegraded_after = 1";
    let gw = Gateway::start_nodes(&[&a, &b], "", canary, "").await;
    all_healthy(&gw, &[&a, &b]).await;
    assert_eq!(agent(&gw, "agent-1").await.status, 200);
    assert_eq!(a.requests(), 1);
    // Its process answers /health, its inference fails: the canary degrades it, and the node is unhealthy.
    a.engine.set_failing(true);
    gw.await_metric(
        "flatshare_gateway_engine_state{node=\"a\",state=\"degraded\"}",
        1.0,
    )
    .await;
    gw.await_metric(&healthy("a"), 0.0).await;
    assert_eq!(agent(&gw, "agent-1").await.status, 200);
    assert_eq!((a.requests(), b.requests()), (1, 1));
    gw.await_metric(&placements("b", "background", "moved"), 1.0)
        .await;
    // A degraded node's canary asks every second, so its recovery shows within a second or two.
    a.engine.set_failing(false);
    gw.await_metric(&healthy("a"), 1.0).await;
}

#[tokio::test]
async fn a_moved_call_that_finds_every_survivor_at_its_max_calls_gets_503() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    let gw = Gateway::start_nodes(&[&a, &b], "max_calls = 1", "", "").await;
    all_healthy(&gw, &[&a, &b]).await;
    assert_eq!(node_of(&gw, "call-1").await, "a");
    assert_eq!(node_of(&gw, "call-2").await, "b");
    a.tts.set_healthy(false);
    gw.await_metric(&healthy("a"), 0.0).await;
    // The survivor's call slot is taken: the moved call's turn, and a new call, are refused with Retry-After.
    unplaced(&turn(&gw, "call-1").await, "no_call_slot");
    unplaced(&place(&gw, "call-4").await, "no_call_slot");
    assert_eq!((a.requests(), b.requests()), (0, 0));
    assert_eq!(turn(&gw, "call-2").await.status, 200);
    // A role no available node takes: no_healthy_node.
    b.tts.set_healthy(false);
    gw.await_metric(&healthy("b"), 0.0).await;
    unplaced(&agent(&gw, "agent-1").await, "no_healthy_node");
    let ready = call(&gw.tcp, "GET", "/ready", &[], "").await;
    assert_eq!(ready.status, 503);
    assert_eq!(ready.body, "no healthy node for realtime\n");
}

#[tokio::test]
async fn agents_keep_to_each_nodes_max_agents_and_one_no_node_fits_gets_503_until_a_slot_frees() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    let toml = format!(
        "{}{}[placement]\nhealth_interval_s = 0.02\nhealth_timeout_s = 0.5\nunhealthy_after = 2\nhealthy_after = 2\n",
        a.toml("max_agents = 2"),
        b.toml("max_agents = 1")
    );
    let config = flatshare_gateway::Config::from_toml(&nodes_toml("", &toml)).unwrap();
    let gw = Gateway::from_config(config, std::sync::Arc::new(flatshare_gateway::LogSink)).await;
    all_healthy(&gw, &[&a, &b]).await;
    // The fewest agents' sessions first, within each node's bound: a, b, a, and then neither has a free slot.
    for session in ["agent-1", "agent-2", "agent-3"] {
        assert_eq!(agent(&gw, session).await.status, 200);
    }
    assert_eq!((a.requests(), b.requests()), (2, 1));
    unplaced(&agent(&gw, "agent-4").await, "no_agent_slot");
    assert_eq!((a.requests(), b.requests()), (2, 1));
    gw.await_metric(
        "flatshare_gateway_placements_refused_total{class=\"background\",reason=\"no_agent_slot\"}",
        1.0,
    )
    .await;
    gw.await_metric(
        "flatshare_gateway_request_outcomes_total{class=\"background\",outcome=\"not_ready\"}",
        1.0,
    )
    .await;
    // A placed agent keeps its node, and a call is no agent: a takes one beside its two agents.
    assert_eq!(agent(&gw, "agent-1").await.status, 200);
    assert_eq!(node_of(&gw, "call-1").await, "a");
    // /nodes shows each node's bound beside its call slots.
    let view = nodes_view(&gw).await;
    let bounds: Vec<serde_json::Value> = view["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| serde_json::json!([node["max_calls"], node["max_agents"]]))
        .collect();
    assert_eq!(
        bounds,
        [serde_json::json!([2, 2]), serde_json::json!([2, 1])]
    );
    assert_eq!(
        view["nodes"][0]["sessions"],
        serde_json::json!({"realtime": 1, "interactive": 0, "background": 2})
    );
    // An agent's session_closed frees its slot, which the refused agent then takes.
    agent_closed(&gw, "agent-2").await;
    assert_eq!(agent(&gw, "agent-4").await.status, 200);
    assert_eq!((a.requests(), b.requests()), (3, 2));
}

#[tokio::test]
async fn with_fill_calls_go_to_the_next_node_only_when_the_first_is_full_or_fails() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    let placement = "[placement]\nhealth_interval_s = 0.02\nhealth_timeout_s = 0.5\nunhealthy_after = 2\n\
                     healthy_after = 2\ncalls = \"fill\"\n";
    let gw = Gateway::start_nodes(&[&a, &b], "", "", placement).await;
    all_healthy(&gw, &[&a, &b]).await;
    // a takes calls up to its max_calls (2), and b, a warm spare, only then.
    assert_eq!(node_of(&gw, "call-1").await, "a");
    assert_eq!(node_of(&gw, "call-2").await, "a");
    assert_eq!(node_of(&gw, "call-3").await, "b");
    // A call's end frees a slot on a, which the next call takes.
    close(&gw, "call-1").await;
    assert_eq!(node_of(&gw, "call-4").await, "a");
    // a fails: its calls move to b while b has call slots, and a call that finds b full, moved or new, is refused as
    // with spread.
    a.tts.set_healthy(false);
    gw.await_metric(&healthy("a"), 0.0).await;
    assert_eq!(node_of(&gw, "call-2").await, "b");
    unplaced(&place(&gw, "call-4").await, "no_call_slot");
    unplaced(&place(&gw, "call-5").await, "no_call_slot");
    // No max_agents: /nodes shows null.
    assert_eq!(
        nodes_view(&gw).await["nodes"][0].get("max_agents"),
        Some(&serde_json::Value::Null)
    );
}

#[tokio::test]
async fn a_node_covering_for_a_failed_one_defers_its_agents_until_the_failed_node_is_healthy() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    let gw = Gateway::start_nodes(&[&a, &b], "", "", "").await;
    all_healthy(&gw, &[&a, &b]).await;
    assert_eq!(node_of(&gw, "call-1").await, "a");
    assert_eq!(agent(&gw, "agent-1").await.status, 200);
    a.tts.set_healthy(false);
    gw.await_metric(&healthy("a"), 0.0).await;
    // The call moves to b, which then covers for a: b's agents wait (failover_background_max_in_flight = 0).
    assert_eq!(turn(&gw, "call-1").await.status, 200);
    gw.await_metric(&covering("b"), 1.0).await;
    let waiting = tokio::spawn({
        let tcp = gw.tcp.clone();
        async move {
            let headers = [AGENTS_AUTH, ("x-session-id", "agent-2")];
            call(&tcp, "POST", "/v1/chat/completions", &headers, CHAT).await
        }
    });
    gw.await_metric("flatshare_gateway_node_failover_waiting{node=\"b\"}", 1.0)
        .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!waiting.is_finished());
    assert_eq!(b.requests(), 1, "only the call's turn reached b");
    // a is healthy again: b covers no more, and the agent's request goes.
    a.tts.set_healthy(true);
    let reply = soon(waiting).await.unwrap();
    assert_eq!(reply.status, 200);
    gw.await_metric(&covering("b"), 0.0).await;
    assert_eq!(b.requests(), 2);
}

#[tokio::test]
async fn an_agent_deferred_past_its_queue_deadline_gets_503_and_a_call_is_never_held_back() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    let top = "[classes.background]\nqueue_timeout_s = 0.3";
    let gw = Gateway::start_nodes(&[&a, &b], "", top, "").await;
    all_healthy(&gw, &[&a, &b]).await;
    assert_eq!(node_of(&gw, "call-1").await, "a");
    a.tts.set_healthy(false);
    gw.await_metric(&healthy("a"), 0.0).await;
    assert_eq!(turn(&gw, "call-1").await.status, 200);
    gw.await_metric(&covering("b"), 1.0).await;
    let reply = agent(&gw, "agent-1").await;
    assert_eq!(reply.status, 503, "{:?}", reply.body);
    assert_eq!(reply.json()["error"]["code"], "queue_timeout");
    assert_eq!(reply.header("retry-after"), Some("1"));
    // Realtime turns go on while the agents wait.
    assert_eq!(node_of(&gw, "call-2").await, "b");
    assert_eq!(turn(&gw, "call-2").await.status, 200);
    assert_eq!(b.requests(), 2);
}

/// A server-sent event stream of two chunks.
fn chunks(node: &str) -> Script {
    Script::Sse {
        chunks: vec![
            Bytes::from(format!("data: {{\"node\":\"{node}\",\"n\":1}}\n\n")),
            Bytes::from_static(b"data: [DONE]\n\n"),
        ],
        gap: Duration::from_millis(10),
    }
}

#[tokio::test]
async fn a_realtime_request_whose_node_fails_before_its_first_content_is_sent_once_more_elsewhere()
{
    let a = TestNode::start("a", chunks("a")).await;
    let b = TestNode::start("b", chunks("b")).await;
    // Pooled connections that idle a moment are not reused: a killed engine's are gone, as a real one's are.
    let top = "upstream_idle_timeout_s = 0.05";
    let gw = Gateway::start_nodes(&[&a, &b], "", top, "").await;
    all_healthy(&gw, &[&a, &b]).await;
    assert_eq!(node_of(&gw, "call-1").await, "a");
    a.engine.kill(true);
    tokio::time::sleep(Duration::from_millis(150)).await;
    // The engine is gone before the turn's first content: the turn goes to b, whole.
    let reply = turn(&gw, "call-1").await;
    assert_eq!(reply.status, 200, "{:?}", reply.body);
    let body = String::from_utf8(reply.body.to_vec()).unwrap();
    assert!(
        body.contains(r#""node":"b""#) && body.ends_with("data: [DONE]\n\n"),
        "{body}"
    );
    assert_eq!((a.requests(), b.requests()), (0, 1));
    gw.await_metric(&placements("b", "realtime", "retried"), 1.0)
        .await;
    gw.await_metric("flatshare_gateway_node_retries_total{node=\"a\"}", 1.0)
        .await;
    // The call lives on b now, and a is passed over while its /health does not answer.
    assert_eq!(node_of(&gw, "call-1").await, "b");
    gw.await_metric("flatshare_gateway_node_ready{node=\"a\"}", 0.0)
        .await;
    assert_eq!(node_of(&gw, "call-2").await, "b");
    // One outcome, completed.
    gw.await_metric(
        "flatshare_gateway_request_outcomes_total{class=\"realtime\",outcome=\"completed\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn a_realtime_stream_that_breaks_after_its_first_content_is_never_sent_again() {
    let broken = Script::Broken {
        content_type: "text/event-stream",
        chunks: vec![Bytes::from_static(b"data: {\"n\":1}\n\n")],
        gap: Duration::from_millis(10),
    };
    let a = TestNode::start("a", broken).await;
    let b = TestNode::start("b", chunks("b")).await;
    let gw = Gateway::start_nodes(&[&a, &b], "", "", "").await;
    all_healthy(&gw, &[&a, &b]).await;
    assert_eq!(node_of(&gw, "call-1").await, "a");
    let headers = [("x-session-id", "call-1")];
    let (response, _connection) =
        open(&gw.unix, "POST", "/v1/chat/completions", &headers, CHAT).await;
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut received = Vec::new();
    let ended_with_error = loop {
        match body.frame().await {
            Some(Ok(frame)) => received.extend_from_slice(&frame.into_data().unwrap()),
            Some(Err(_)) => break true,
            None => break false,
        }
    };
    let text = String::from_utf8(received).unwrap();
    // The first content, then the error event of vLLM's streams, then an abort: never b's reply after a's.
    assert!(ended_with_error, "{text}");
    assert!(text.starts_with("data: {\"n\":1}\n\n"), "{text}");
    assert!(
        text.contains("upstream_unavailable") && !text.contains("[DONE]"),
        "{text}"
    );
    assert_eq!(b.requests(), 0);
    gw.await_metric(
        "flatshare_gateway_request_outcomes_total{class=\"realtime\",outcome=\"transport_error\"}",
        1.0,
    )
    .await;
    assert_eq!(
        metric(
            &gw.metrics().await,
            "flatshare_gateway_node_retries_total{node=\"a\"}"
        ),
        Some(0.0)
    );
}

#[tokio::test]
async fn ready_and_the_engines_metrics_name_the_nodes() {
    let a = TestNode::start("a", Script::Json("{}")).await;
    let b = TestNode::start("b", Script::Json("{}")).await;
    let toml = format!(
        "{}{}[placement]\nhealth_interval_s = 0.02\nunhealthy_after = 2\nhealthy_after = 2\n",
        a.toml(""),
        b.toml("roles = [\"background\"]")
    );
    let config = flatshare_gateway::Config::from_toml(&nodes_toml("", &toml)).unwrap();
    let gw = Gateway::from_config(config, std::sync::Arc::new(flatshare_gateway::LogSink)).await;
    all_healthy(&gw, &[&a, &b]).await;
    let ready = call(&gw.tcp, "GET", "/ready", &[], "").await;
    assert_eq!(
        (ready.status.as_u16(), &ready.body[..]),
        (200, &b"ready\n"[..])
    );
    let named = call(&gw.unix, "GET", "/metrics/engine?node=b", &[], "").await;
    assert_eq!(named.status, 200);
    assert_eq!(named.body, ENGINE_METRICS);
    let none = call(&gw.unix, "GET", "/metrics/engine", &[], "").await;
    assert_eq!(none.status, 400);
    assert_eq!(none.json()["error"]["code"], "node_required");
    let unknown = call(&gw.unix, "GET", "/metrics/engine?node=c", &[], "").await;
    assert_eq!(unknown.status, 404);
    // b takes agents only: calls go to a, whatever b holds.
    assert_eq!(node_of(&gw, "call-1").await, "a");
    assert_eq!(node_of(&gw, "call-2").await, "a");
    // /nodes needs the metrics key on the TCP listener, as /metrics does.
    assert_eq!(call(&gw.tcp, "GET", "/nodes", &[], "").await.status, 401);
    let view = call(&gw.tcp, "GET", "/nodes", &[METRICS_AUTH], "").await;
    assert_eq!(view.status, 200);
    assert_eq!(
        view.json()["nodes"][1]["roles"],
        serde_json::json!(["background"])
    );
}
