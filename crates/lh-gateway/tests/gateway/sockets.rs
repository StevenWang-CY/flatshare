//! Listening sockets the gateway inherits, as systemd's socket activation passes them (`LISTEN_FDS`), so that they
//! outlive any one gateway process. `reload.rs` runs the binary on them.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::net::UnixListener;
use std::sync::Arc;
use std::time::Duration;

use lh_gateway::{Config, Inherited, LogSink, Server};

use crate::common::*;

/// A chat completion of the agents tenant on its own connection, which closes after the answer.
fn raw_request() -> Vec<u8> {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer {AGENTS_KEY}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{CHAT}",
        CHAT.len()
    )
    .into_bytes()
}

/// Connects to `addr` now and sends a request, before any gateway accepts there; the thread gives the answer.
pub fn early_client(addr: SocketAddr) -> std::thread::JoinHandle<String> {
    let mut stream = TcpStream::connect(addr).expect("a listening socket queues the connection");
    stream.write_all(&raw_request()).unwrap();
    std::thread::spawn(move || {
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).unwrap();
        answer
    })
}

#[tokio::test]
async fn inherited_sockets_serve_the_connections_made_before_and_stay_after_the_stop() {
    let mock = Mock::start(Script::Json("{}")).await;
    let tcp = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = tcp.local_addr().unwrap();
    let path = socket_path();
    let unix = UnixListener::bind(&path).unwrap();
    let early = early_client(addr);
    let config = Config::from_toml(&listeners_toml(&mock.upstream, addr, &path, "")).unwrap();
    let mut gw = Gateway::from_inherited(config, Inherited::default().tcp(tcp).unix(unix)).await;
    let answer = tokio::task::spawn_blocking(move || early.join().unwrap())
        .await
        .unwrap();
    assert!(answer.starts_with("HTTP/1.1 200 OK\r\n"), "{answer}");
    let reply = call(&gw.unix, "POST", "/v1/chat/completions", &[], CHAT).await;
    assert_eq!(reply.status, 200);
    gw.stopped().await;
    assert!(
        path.exists(),
        "an inherited socket's file is its owner's to remove"
    );
    std::fs::remove_file(&path).unwrap();
}

#[tokio::test]
async fn a_listener_without_an_inherited_socket_is_bound_and_an_unnamed_socket_is_refused() {
    let mock = Mock::start(Script::Json("{}")).await;
    let path = socket_path();
    let unix = UnixListener::bind(&path).unwrap();
    // The TCP listener is the gateway's own (127.0.0.1:0); the Unix one is inherited.
    let config = Config::from_toml(&gateway_toml(mock.addr, &path, "")).unwrap();
    let mut gw = Gateway::from_inherited(config.clone(), Inherited::default().unix(unix)).await;
    assert_eq!(call(&gw.tcp, "GET", "/health", &[], "").await.status, 200);
    assert_eq!(call(&gw.unix, "GET", "/health", &[], "").await.status, 200);
    gw.stopped().await;

    // A socket at a path no entry names. (A TCP one on 127.0.0.1 would do for the entry's port 0, any port.)
    let stray_path = socket_path();
    let stray = UnixListener::bind(&stray_path).unwrap();
    let err = Server::bind_with(config, Arc::new(LogSink), Inherited::default().unix(stray))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains(&format!(
            "inherited sockets that no [[listen]] entry names, so that nothing would accept on them: unix {}",
            stray_path.display()
        )),
        "{err}"
    );
    // The refused start bound nothing: the socket file the first gateway left is as it was.
    assert!(path.exists());
    for file in [&path, &stray_path] {
        std::fs::remove_file(file).unwrap();
    }
}
