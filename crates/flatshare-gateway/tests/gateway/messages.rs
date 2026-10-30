//! The Messages API (`/v1/messages` and `/v1/messages/count_tokens`), as Claude Code sends it: the same tenants,
//! classes, sessions, sanitization and caps as Chat Completions, with the reply cap on `max_tokens`.

use std::time::Duration;

use bytes::Bytes;
use serde_json::json;

use crate::common::*;

const MESSAGES: &str =
    r#"{"model":"qwen","max_tokens":200,"messages":[{"role":"user","content":"hi"}]}"#;

#[tokio::test]
async fn messages_take_the_tenants_classes_and_sessions_of_chat_and_lose_the_engines_inputs() {
    let mock = Mock::start(Script::Json(r#"{"id":"msg_1","type":"message"}"#)).await;
    let gw = Gateway::start(mock.addr, "").await;
    let unauthorized = call(&gw.tcp, "POST", "/v1/messages", &[], MESSAGES).await;
    assert_eq!(unauthorized.status, 401);
    assert_eq!(unauthorized.json()["error"]["code"], "invalid_api_key");

    let body = r#"{"model":"qwen","max_tokens":200,"messages":[{"role":"user","content":"hi"}],"priority":-100,
        "kv_transfer_params":{"a":1},"ec_transfer_params":{"b":2},"cache_salt":"theirs","vllm_xargs":{"session_id":"s"}}"#;
    let claude = ("x-claude-code-session-id", "cc-session-1");
    let headers = [
        BENCH_AUTH,
        ("x-vllm-priority", "-100"),
        ("x-flatshare-class", "realtime"),
        claude,
    ];
    let reply = call(&gw.tcp, "POST", "/v1/messages", &headers, body).await;
    assert_eq!(reply.status, 200);
    assert_eq!(&reply.body[..], br#"{"id":"msg_1","type":"message"}"#);
    let seen = mock.last();
    assert_eq!(seen.path, "/v1/messages");
    // An untrusted tenant's class header is ignored: its default, interactive, is priority 1.
    assert_eq!(seen.header("x-vllm-priority"), Some("1"));
    assert_eq!(seen.header("x-flatshare-class"), None);
    assert_eq!(
        seen.json(),
        json!({"model": "qwen", "max_tokens": 200, "messages": [{"role": "user", "content": "hi"}],
            "cache_salt": BENCH_SALT})
    );
    // The session is Claude Code's, as a chat request of the same header gets it.
    let session = seen.header("x-session-id").unwrap().to_owned();
    assert_eq!(
        reply.header("x-flatshare-session-id"),
        Some(session.as_str())
    );
    let chat = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[BENCH_AUTH, claude],
        CHAT,
    )
    .await;
    assert_eq!(chat.status, 200);
    assert_eq!(mock.last().header("x-session-id"), Some(session.as_str()));

    // A trusted tenant chooses its class within its bound: agents may ask for interactive.
    let agents = [AGENTS_AUTH, ("x-flatshare-class", "interactive")];
    assert_eq!(
        call(&gw.tcp, "POST", "/v1/messages", &agents, MESSAGES)
            .await
            .status,
        200
    );
    assert_eq!(mock.last().header("x-vllm-priority"), Some("1"));
    assert_eq!(
        call(&gw.tcp, "POST", "/v1/messages", &[AGENTS_AUTH], MESSAGES)
            .await
            .status,
        200
    );
    assert_eq!(mock.last().header("x-vllm-priority"), Some("2"));
}

#[tokio::test]
async fn a_capped_class_gets_its_messages_reply_cap_on_max_tokens() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "[classes.realtime]\nmax_reply_tokens = 24").await;
    let body = |limits: &str| {
        format!(r#"{{"model":"qwen","messages":[{{"role":"user","content":"hi"}}]{limits}}}"#)
    };
    // The voice tenant's realtime replies: a longer limit lowered to the cap, a shorter one kept, none given the cap,
    // and another route's limit field, which vLLM's Messages model ignores, gone.
    for (asked, sent) in [
        (r#","max_tokens":200"#, 24),
        (r#","max_tokens":8"#, 8),
        ("", 24),
        (r#","max_tokens":12,"max_completion_tokens":4000"#, 12),
    ] {
        let reply = call(&gw.unix, "POST", "/v1/messages", &[], &body(asked)).await;
        assert_eq!(reply.status, 200);
        let seen = mock.last().json();
        assert_eq!(seen["max_tokens"], sent, "{seen}");
        assert!(seen.get("max_completion_tokens").is_none(), "{seen}");
    }
    // An interactive reply is not capped.
    let reply = call(
        &gw.tcp,
        "POST",
        "/v1/messages",
        &[BENCH_AUTH],
        &body(r#","max_tokens":200"#),
    )
    .await;
    assert_eq!(reply.status, 200);
    assert_eq!(mock.last().json()["max_tokens"], 200);
    // A limit vLLM would read otherwise is refused, as on chat.
    let twice = body(r#","max_tokens":1,"max_tokens":9000"#);
    let reply = call(&gw.unix, "POST", "/v1/messages", &[], &twice).await;
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["code"], "duplicate_field");
}

#[tokio::test]
async fn a_streamed_messages_reply_is_relayed_byte_for_byte() {
    let chunks: Vec<Bytes> = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .map(|chunk: &'static str| Bytes::from_static(chunk.as_bytes()))
    .to_vec();
    let expected = chunks.concat();
    let mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(10),
    })
    .await;
    let gw = Gateway::start(mock.addr, "").await;
    let body = r#"{"model":"qwen","max_tokens":20,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let reply = call(&gw.tcp, "POST", "/v1/messages", &[AGENTS_AUTH], body).await;
    assert_eq!(reply.status, 200);
    assert_eq!(&reply.body[..], &expected[..]);
    assert_eq!(mock.last().json()["stream"], true);
}

#[tokio::test]
async fn count_tokens_passes_through_under_the_same_contract_and_no_cap() {
    let mock = Mock::start(Script::Json(r#"{"input_tokens":7}"#)).await;
    let gw = Gateway::start(mock.addr, "[classes.realtime]\nmax_reply_tokens = 24").await;
    let body = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#;
    assert_eq!(
        call(&gw.tcp, "POST", "/v1/messages/count_tokens", &[], body)
            .await
            .status,
        401
    );
    let reply = call(&gw.unix, "POST", "/v1/messages/count_tokens", &[], body).await;
    assert_eq!(reply.status, 200);
    assert_eq!(&reply.body[..], br#"{"input_tokens":7}"#);
    let seen = mock.last();
    assert_eq!(seen.path, "/v1/messages/count_tokens");
    assert_eq!(seen.header("x-vllm-priority"), Some("0"));
    // It generates nothing, so the realtime cap writes no limit.
    assert_eq!(
        seen.json(),
        json!({"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "cache_salt": VOICE_SALT})
    );
    let metrics = gw.metrics().await;
    assert!(
        metrics.contains(r#"route="/v1/messages/count_tokens""#),
        "{metrics}"
    );
}
