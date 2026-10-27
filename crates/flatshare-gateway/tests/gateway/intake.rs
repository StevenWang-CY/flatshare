//! The intake: requests in flight in all, by class and by tenant, refused before their bodies are read; the budget of
//! the bodies the gateway holds, taken as their bytes arrive; and the bound on the requests waiting at the admission
//! point, in all its queues.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use flatshare_gateway::{Config, LogSink};

use crate::common::*;

/// What a raw client got: the answer's head and body as text, and how long after its head was sent.
struct Answer {
    text: String,
    after: Duration,
}

impl Answer {
    fn status(&self) -> &str {
        self.text.get(9..12).unwrap_or_default()
    }

    fn header(&self, name: &str) -> Option<&str> {
        let head = self.text.split("\r\n\r\n").next()?;
        head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }

    fn json(&self) -> serde_json::Value {
        let body = self
            .text
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .unwrap();
        serde_json::from_str(body).unwrap()
    }
}

/// Sends a request's `head` on a new connection, then each part of its body after its pause, and reads the answer
/// for up to five seconds: a client whose body never comes, or comes slowly.
fn raw(
    addr: SocketAddr,
    head: String,
    parts: Vec<(Duration, Vec<u8>)>,
) -> std::thread::JoinHandle<Answer> {
    std::thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let sent = Instant::now();
        stream.write_all(head.as_bytes()).unwrap();
        for (pause, part) in parts {
            std::thread::sleep(pause);
            // The gateway may have answered and closed already.
            if stream.write_all(&part).is_err() {
                break;
            }
        }
        let mut bytes = Vec::new();
        let mut buf = [0; 4096];
        loop {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => bytes.extend_from_slice(&buf[..n]),
            }
            // The whole answer: its head, and the body its Content-Length gives.
            let text = String::from_utf8_lossy(&bytes);
            if let Some((head, body)) = text.split_once("\r\n\r\n") {
                let length = head.lines().find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())?
                });
                if length.is_some_and(|length| body.len() >= length) {
                    break;
                }
            }
        }
        Answer {
            text: String::from_utf8_lossy(&bytes).into_owned(),
            after: sent.elapsed(),
        }
    })
}

/// A chat completion's head on the TCP listener, with `auth` and `framing` (its Content-Length or chunked encoding).
fn head(auth: &str, framing: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer {auth}\r\n\
         Content-Type: application/json\r\n{framing}\r\n\r\n"
    )
}

fn tcp(gw: &Gateway) -> SocketAddr {
    match gw.tcp {
        Target::Tcp(addr) => addr,
        Target::Unix(_) => unreachable!("the TCP listener"),
    }
}

async fn answered(client: std::thread::JoinHandle<Answer>) -> Answer {
    tokio::task::spawn_blocking(move || client.join().unwrap())
        .await
        .unwrap()
}

/// The answer a refused request gets: its status, its error's code, and `Retry-After: 1`.
fn refused(answer: &Answer, status: &str, code: &str) {
    assert_eq!(answer.status(), status, "{}", answer.text);
    assert_eq!(answer.json()["error"]["code"], code, "{}", answer.text);
    assert_eq!(answer.header("retry-after"), Some("1"), "{}", answer.text);
    assert!(
        answer.after < Duration::from_secs(1),
        "answered after {:?}",
        answer.after
    );
}

const IN_FLIGHT: &str = "flatshare_gateway_in_flight{class=\"background\"}";

#[tokio::test]
async fn a_full_gateway_refuses_a_request_before_it_reads_its_body() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "max_in_flight = 1").await;
    let held = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (TAG, "held"), (GATE, "1")], CHAT);
    mock.await_requests(1).await;
    gw.await_metric(IN_FLIGHT, 1.0).await;
    // Bodies that never come, declared or chunked: each is refused at once, never read.
    for framing in ["Content-Length: 100000", "Transfer-Encoding: chunked"] {
        let answer = answered(raw(tcp(&gw), head(AGENTS_KEY, framing), Vec::new())).await;
        refused(&answer, "503", "in_flight_limit");
        assert_eq!(answer.json()["error"]["type"], "server_error");
    }
    // Every class is refused alike, and no refused request reached the engine.
    let voice = call(&gw.unix, "POST", "/v1/chat/completions", &[], CHAT).await;
    assert_eq!(voice.status, 503);
    assert_eq!(mock.requests().len(), 1);
    mock.release(1);
    assert_eq!(held.await.unwrap().status, 200);
    gw.await_metric(IN_FLIGHT, 0.0).await;
    assert_eq!(
        call(&gw.unix, "POST", "/v1/chat/completions", &[], CHAT)
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn a_full_class_is_refused_and_the_others_go_on() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "[classes.background]\nmax_in_flight = 1").await;
    let held = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (GATE, "1")], CHAT);
    mock.await_requests(1).await;
    let answer = answered(raw(
        tcp(&gw),
        head(AGENTS_KEY, "Content-Length: 100"),
        Vec::new(),
    ))
    .await;
    refused(&answer, "503", "class_in_flight_limit");
    // Realtime and interactive requests are not background's.
    assert_eq!(
        call(&gw.unix, "POST", "/v1/chat/completions", &[], CHAT)
            .await
            .status,
        200
    );
    let interactive = call(&gw.tcp, "POST", "/v1/chat/completions", &[BENCH_AUTH], CHAT).await;
    assert_eq!(interactive.status, 200);
    mock.release(1);
    assert_eq!(held.await.unwrap().status, 200);
}

#[tokio::test]
async fn a_tenant_at_its_bound_gets_429_and_the_others_are_served() {
    let mock = Mock::start(Script::Json("{}")).await;
    let toml = gateway_toml(mock.addr, &socket_path(), "")
        .replace("[tenants.bench]\n", "[tenants.bench]\nmax_in_flight = 1\n");
    let gw = Gateway::from_config(Config::from_toml(&toml).unwrap(), Arc::new(LogSink)).await;
    let held = spawn_call(&gw.tcp, vec![BENCH_AUTH, (GATE, "1")], CHAT);
    mock.await_requests(1).await;
    let answer = answered(raw(
        tcp(&gw),
        head(BENCH_KEY, "Content-Length: 100"),
        Vec::new(),
    ))
    .await;
    refused(&answer, "429", "tenant_in_flight_limit");
    assert_eq!(answer.json()["error"]["type"], "rate_limit_error");
    // Another tenant's request goes up while bench's waits, and bench's next after its own ends.
    let agents = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    assert_eq!(agents.status, 200);
    mock.release(1);
    assert_eq!(held.await.unwrap().status, 200);
    let bench = call(&gw.tcp, "POST", "/v1/chat/completions", &[BENCH_AUTH], CHAT).await;
    assert_eq!(bench.status, 200);
}

/// A chat completion of `n` bytes, padded.
fn padded(n: usize) -> String {
    let empty = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}],"pad":""}"#;
    format!(
        r#"{{"model":"qwen","messages":[{{"role":"user","content":"hi"}}],"pad":"{}"}}"#,
        "x".repeat(n - empty.len())
    )
}

#[tokio::test]
async fn bodies_take_their_bytes_from_one_budget_as_they_arrive() {
    let mock = Mock::start(Script::Json("{}")).await;
    let top = "max_body_bytes = 1000\nmax_buffered_body_bytes = 1000";
    let gw = Gateway::start(mock.addr, top).await;
    let six_hundred: &'static str = Box::leak(padded(600).into_boxed_str());
    let held = spawn_call(&gw.tcp, vec![AGENTS_AUTH, (GATE, "1")], six_hundred);
    mock.await_requests(1).await;
    let bytes = "flatshare_gateway_buffered_body_bytes";
    gw.await_metric(bytes, 600.0).await;
    // A declared length the budget cannot hold is refused before a byte of it is read.
    let answer = answered(raw(
        tcp(&gw),
        head(AGENTS_KEY, "Content-Length: 600"),
        Vec::new(),
    ))
    .await;
    refused(&answer, "503", "body_budget_exhausted");
    // A chunked body takes its bytes as they come: 300 fit beside the 600 held, 300 more do not.
    let chunk = |n: usize| format!("{n:x}\r\n{}\r\n", "y".repeat(n)).into_bytes();
    let parts = vec![
        (Duration::ZERO, chunk(300)),
        (Duration::from_millis(200), chunk(300)),
    ];
    let answer = answered(raw(
        tcp(&gw),
        head(AGENTS_KEY, "Transfer-Encoding: chunked"),
        parts,
    ))
    .await;
    assert_eq!(answer.status(), "503", "{}", answer.text);
    assert_eq!(answer.json()["error"]["code"], "body_budget_exhausted");
    assert!(
        answer.after >= Duration::from_millis(200),
        "after the second chunk"
    );
    gw.await_metric(bytes, 600.0).await;
    // The held request's bytes come back when it ends, and a body of the same size fits again.
    mock.release(1);
    assert_eq!(held.await.unwrap().status, 200);
    gw.await_metric(bytes, 0.0).await;
    let reply = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        six_hundred,
    )
    .await;
    assert_eq!(reply.status, 200);
    gw.await_metric(bytes, 0.0).await;
}

#[tokio::test]
async fn the_admission_queues_hold_at_most_max_queued_requests_in_all() {
    let mock = Mock::start(Script::Json("{}")).await;
    let top = "max_queued = 2\n[admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 1";
    let gw = Gateway::start(mock.addr, top).await;
    let background = |tag| vec![AGENTS_AUTH, (TAG, tag), (GATE, "1")];
    let mut replies = vec![spawn_call(&gw.tcp, background("bg1"), CHAT)];
    mock.await_requests(1).await;
    for (n, tag) in ["bg2", "bg3"].into_iter().enumerate() {
        replies.push(spawn_call(&gw.tcp, background(tag), CHAT));
        gw.await_metric(
            "flatshare_gateway_admission_waiting{class=\"background\"}",
            n as f64 + 1.0,
        )
        .await;
    }
    // The class's queue holds 1,024; the queues together hold max_queued.
    let full = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &background("bg4"),
        CHAT,
    )
    .await;
    assert_eq!(full.status, 429);
    let error = &full.json()["error"];
    assert_eq!(
        (&error["code"], &error["type"]),
        (&"queue_full".into(), &"rate_limit_error".into())
    );
    assert!(
        error["message"].as_str().unwrap().contains("max_queued"),
        "{error}"
    );
    assert_eq!(full.header("retry-after"), Some("1"));
    gw.await_metric(
        "flatshare_gateway_admission_refused_total{class=\"background\",reason=\"max_queued\"}",
        1.0,
    )
    .await;
    mock.release(3);
    for reply in replies {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    assert_eq!(mock.tags(), ["bg1", "bg2", "bg3"]);
}
