//! The binary as its systemd unit runs it: on the listening sockets of `lh-gateway.socket` (`LISTEN_FDS`, from fd 3
//! on), telling the manager when it is ready (`NOTIFY_SOCKET`), and reloading on SIGHUP, which hands the sockets to
//! a successor.

use std::net::{SocketAddr, TcpListener};
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use bytes::Bytes;
use command_fds::{CommandFdExt, FdMapping};
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use tokio::net::{TcpStream, UnixDatagram, UnixStream};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::common::*;
use crate::sockets::early_client;

/// What systemd provides the gateway's unit: the sockets of `lh-gateway.socket`, its configuration file, and the
/// manager's end of `NOTIFY_SOCKET`.
struct Unit {
    tcp: SocketAddr,
    socket: PathBuf,
    config: PathBuf,
    manager: UnixDatagram,
    manager_path: PathBuf,
}

impl Unit {
    /// Binds the sockets, as the socket unit does, and writes the configuration, with `top`.
    fn new(upstream: &str, top: &str) -> (Self, TcpListener, UnixListener) {
        let tcp = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = tcp.local_addr().unwrap();
        let socket = socket_path();
        let unix = UnixListener::bind(&socket).unwrap();
        let name = socket_path().with_extension("toml");
        let config = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name.file_name().unwrap());
        std::fs::write(&config, listeners_toml(upstream, addr, &socket, top)).unwrap();
        let manager_path = socket_path();
        let manager = UnixDatagram::bind(&manager_path).unwrap();
        let unit = Self {
            tcp: addr,
            socket,
            config,
            manager,
            manager_path,
        };
        (unit, tcp, unix)
    }

    /// Starts the gateway on the sockets, as systemd starts the service.
    fn start(&self, tcp: TcpListener, unix: UnixListener) -> Child {
        let mut command = Command::new(env!("CARGO_BIN_EXE_lh-gateway"));
        command
            .arg("--config")
            .arg(&self.config)
            .env("LISTEN_FDS", "2")
            .env_remove("LISTEN_PID")
            .env("NOTIFY_SOCKET", &self.manager_path)
            .stdout(Stdio::null())
            .fd_mappings(vec![
                FdMapping {
                    parent_fd: OwnedFd::from(tcp),
                    child_fd: 3,
                },
                FdMapping {
                    parent_fd: OwnedFd::from(unix),
                    child_fd: 4,
                },
            ])
            .unwrap();
        // The command, which holds the test's copies of the sockets, goes with this scope.
        command.spawn().unwrap()
    }

    /// The next notification the manager gets, if one comes within `within`.
    async fn notified(&self, within: Duration) -> Option<String> {
        let mut buf = [0; 512];
        let n = tokio::time::timeout(within, self.manager.recv(&mut buf))
            .await
            .ok()?
            .unwrap();
        Some(String::from_utf8(buf[..n].to_vec()).unwrap())
    }

    fn targets(&self) -> [Target; 2] {
        [Target::Tcp(self.tcp), Target::Unix(self.socket.clone())]
    }
}

impl Drop for Unit {
    fn drop(&mut self) {
        for file in [&self.socket, &self.config, &self.manager_path] {
            let _ = std::fs::remove_file(file);
        }
    }
}

/// A gateway process, killed when the test ends if it still runs.
struct Process(u32);

impl Process {
    fn signal(&self, signal: Signal) {
        kill(pid(self.0), signal).unwrap();
    }

    fn alive(&self) -> bool {
        kill(pid(self.0), None).is_ok()
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = kill(pid(self.0), Signal::SIGKILL);
    }
}

fn pid(pid: u32) -> Pid {
    Pid::from_raw(i32::try_from(pid).unwrap())
}

async fn wait(child: Child) -> std::process::ExitStatus {
    let mut child = child;
    tokio::task::spawn_blocking(move || child.wait().unwrap())
        .await
        .unwrap()
}

/// `GET /v1/models` on a new connection: its status, or why it got none (a refused connection among them).
async fn try_models(target: &Target) -> Result<u16, String> {
    let request = || {
        http::Request::get("/v1/models")
            .header("host", "gateway")
            .header("authorization", "Bearer bench-key")
            .body(Full::new(Bytes::new()))
            .unwrap()
    };
    let status = match target {
        Target::Tcp(addr) => {
            let stream = TcpStream::connect(addr).await.map_err(|e| e.to_string())?;
            send(TokioIo::new(stream), request()).await?
        }
        Target::Unix(path) => {
            let stream = UnixStream::connect(path).await.map_err(|e| e.to_string())?;
            send(TokioIo::new(stream), request()).await?
        }
    };
    Ok(status)
}

async fn send<I>(io: TokioIo<I>, request: http::Request<Full<Bytes>>) -> Result<u16, String>
where
    I: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut sender, connection) = http1::handshake(io).await.map_err(|e| e.to_string())?;
    tokio::spawn(connection);
    let response = sender
        .send_request(request)
        .await
        .map_err(|e| e.to_string())?;
    let status = response.status().as_u16();
    response
        .into_body()
        .collect()
        .await
        .map_err(|e| e.to_string())?;
    Ok(status)
}

/// A request on a new connection every 20 ms, on the TCP port and the Unix socket in turn, until `stop`: what
/// each got, with when it was sent.
fn probe(
    targets: [Target; 2],
    mut stop: watch::Receiver<bool>,
) -> JoinHandle<Vec<(Instant, Result<u16, String>)>> {
    tokio::spawn(async move {
        let mut seen = Vec::new();
        for target in targets.iter().cycle() {
            if *stop.borrow() {
                break;
            }
            let sent = Instant::now();
            seen.push((sent, try_models(target).await));
            tokio::select! {
                _ = stop.changed() => {}
                () = tokio::time::sleep(Duration::from_millis(20)) => {}
            }
        }
        seen
    })
}

#[tokio::test]
async fn the_binary_serves_the_sockets_systemd_passes_and_says_when_it_is_ready() {
    let mock = Mock::start(Script::Json("{}")).await;
    let (unit, tcp, unix) = Unit::new(&mock.upstream, "");
    let early = early_client(unit.tcp);
    let child = unit.start(tcp, unix);
    let gateway = Process(child.id());
    assert_eq!(
        unit.notified(Duration::from_secs(10)).await.as_deref(),
        Some("READY=1")
    );
    let answer = tokio::task::spawn_blocking(move || early.join().unwrap())
        .await
        .unwrap();
    assert!(answer.starts_with("HTTP/1.1 200 OK\r\n"), "{answer}");
    let voice = Target::Unix(unit.socket.clone());
    let reply = call(&voice, "POST", "/v1/chat/completions", &[], CHAT).await;
    assert_eq!(reply.status, 200);
    assert_eq!(
        mock.last().header("x-vllm-priority"),
        Some("0"),
        "the voice socket's tenant"
    );

    gateway.signal(Signal::SIGTERM);
    let status = wait(child).await;
    assert!(status.success(), "{status}");
    assert!(unit.socket.exists(), "the socket unit's file stays");
}

#[tokio::test]
async fn a_reload_hands_the_sockets_to_a_successor_and_refuses_no_request() {
    let chunks: Vec<Bytes> = (0..40)
        .map(|i| Bytes::from(format!("data: {i}\n\n")))
        .collect();
    let whole: Vec<u8> = chunks.concat();
    let mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(50),
    })
    .await;
    let (unit, tcp, unix) = Unit::new(
        &mock.upstream,
        "drain_timeout_s = 10\nbackground_drain_s = 0.3",
    );
    let child = unit.start(tcp, unix);
    let first = Process(child.id());
    assert_eq!(
        unit.notified(Duration::from_secs(10)).await.as_deref(),
        Some("READY=1")
    );
    // A voice turn and an agent turn in flight, and new requests on both sockets all along.
    let [tcp_target, unix_target] = unit.targets();
    let realtime = begin(&unix_target, &[]).await;
    let background = begin(&tcp_target, &[AGENTS_AUTH]).await;
    let (stop, stopping) = watch::channel(false);
    let probes = probe(unit.targets(), stopping);
    tokio::time::sleep(Duration::from_millis(100)).await;

    let reloaded = Instant::now();
    first.signal(Signal::SIGHUP);
    let notified = unit
        .notified(Duration::from_secs(10))
        .await
        .expect("the successor's notification");
    let second: u32 = notified
        .strip_prefix("MAINPID=")
        .and_then(|rest| rest.strip_suffix("\nREADY=1"))
        .unwrap_or_else(|| panic!("{notified:?}"))
        .parse()
        .unwrap();
    let second = Process(second);
    assert_ne!(second.0, first.0);
    let took_over = reloaded.elapsed();
    assert!(
        took_over < Duration::from_secs(5),
        "took over {took_over:?} after SIGHUP"
    );

    // The first process drains: the agent turn is cut at its grace, the voice turn finishes whole, and it exits 0.
    let (bytes, aborted, _) = soon(finish(background)).await.unwrap();
    assert!(aborted);
    assert!(
        String::from_utf8_lossy(&bytes).contains("\"type\":\"draining\""),
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let (bytes, aborted, _) = soon(finish(realtime)).await.unwrap();
    assert!(!aborted);
    assert_eq!(bytes, whole);
    let status = soon(wait(child)).await;
    assert!(status.success(), "{status}");

    // The successor serves on alone, and no request found the sockets closed or got an error.
    tokio::time::sleep(Duration::from_millis(300)).await;
    stop.send(true).unwrap();
    let seen = probes.await.unwrap();
    let failed: Vec<_> = seen.iter().filter(|(_, got)| *got != Ok(200)).collect();
    assert!(failed.is_empty(), "{failed:?}");
    let after = seen
        .iter()
        .filter(|(sent, _)| *sent > reloaded + took_over)
        .count();
    assert!(
        seen.len() > 20 && after > 10,
        "{} probes, {after} after the takeover",
        seen.len()
    );
    assert!(second.alive());
    assert!(unit.socket.exists(), "the sockets stay for the successor");

    second.signal(Signal::SIGTERM);
    let deadline = Instant::now() + Duration::from_secs(5);
    while second.alive() {
        assert!(Instant::now() < deadline, "the successor stops on SIGTERM");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_reload_whose_successor_cannot_start_changes_nothing() {
    let mock = Mock::start(Script::Json("{}")).await;
    let (unit, tcp, unix) = Unit::new(&mock.upstream, "");
    let mut child = unit.start(tcp, unix);
    let gateway = Process(child.id());
    assert_eq!(
        unit.notified(Duration::from_secs(10)).await.as_deref(),
        Some("READY=1")
    );
    // The configuration on disk is broken now: the successor exits 2 at once.
    std::fs::write(&unit.config, "nonsense = true\n").unwrap();
    gateway.signal(Signal::SIGHUP);
    assert_eq!(unit.notified(Duration::from_secs(2)).await, None);
    for target in unit.targets() {
        assert_eq!(try_models(&target).await, Ok(200));
    }
    assert!(child.try_wait().unwrap().is_none(), "the gateway serves on");

    gateway.signal(Signal::SIGTERM);
    let status = wait(child).await;
    assert!(status.success(), "{status}");
}

#[tokio::test]
async fn a_gateway_that_bound_its_own_sockets_leaves_them_to_its_successor() {
    let mock = Mock::start(Script::Json("{}")).await;
    let socket = socket_path();
    let (unit, _, _) = Unit::new(&mock.upstream, "");
    // Its own TCP port (0, any) and socket file, as when no socket unit runs it.
    std::fs::write(&unit.config, gateway_toml(mock.addr, &socket, "")).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_lh-gateway"));
    command
        .arg("--config")
        .arg(&unit.config)
        .env_remove("LISTEN_FDS")
        .env("NOTIFY_SOCKET", &unit.manager_path)
        .stdout(Stdio::null());
    let child = command.spawn().unwrap();
    let first = Process(child.id());
    assert_eq!(
        unit.notified(Duration::from_secs(10)).await.as_deref(),
        Some("READY=1")
    );
    first.signal(Signal::SIGHUP);
    let notified = unit.notified(Duration::from_secs(10)).await.unwrap();
    let second = Process(
        notified
            .strip_prefix("MAINPID=")
            .and_then(|rest| rest.strip_suffix("\nREADY=1"))
            .unwrap()
            .parse()
            .unwrap(),
    );
    let status = soon(wait(child)).await;
    assert!(status.success(), "{status}");
    // The file the first one bound stays for the successor, which answers on it.
    assert!(socket.exists());
    assert_eq!(try_models(&Target::Unix(socket.clone())).await, Ok(200));
    second.signal(Signal::SIGTERM);
    let deadline = Instant::now() + Duration::from_secs(5);
    while second.alive() {
        assert!(Instant::now() < deadline, "the successor stops on SIGTERM");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let _ = std::fs::remove_file(&socket);
}
