//! The test harness: a mock engine, a gateway on ephemeral listeners, and a raw HTTP/1.1 client.

use std::convert::Infallible;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use bytes::Bytes;
use flatshare_gateway::{
    AcceptedEvent, Config, EventSink, Inherited, ListenAddr, LogSink, MetricsReader, Server,
};
use futures_util::stream;
use http::header::CONTENT_TYPE;
use http::{HeaderMap, Method, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use sha2::{Digest, Sha256};
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio::task::JoinHandle;

pub const AGENTS_KEY: &str = "agents-key";
pub const BENCH_KEY: &str = "bench-key";

/// A request header naming the request, for [`Mock::tags`] and [`Mock::dropped`].
pub const TAG: &str = "x-mock-tag";
/// A request header that makes the mock engine hold its answer until [`Mock::release`]: a JSON response's
/// headers, or an event stream's first chunk, as an engine holds them through a prefill.
pub const GATE: &str = "x-mock-gate";

/// The mock engine's `/metrics`, as vLLM's exposition starts, with vLLM's content type.
pub const ENGINE_METRICS: &str = "# HELP vllm:num_requests_running Number of requests in model execution batches.\n\
    # TYPE vllm:num_requests_running gauge\n\
    vllm:num_requests_running{engine=\"0\",model_name=\"qwen\"} 0.0\n";
pub const ENGINE_METRICS_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// The metrics key of every test gateway, which `GET /metrics` asks for on its TCP listener.
pub const METRICS_KEY: &str = "metrics-key";
pub const METRICS_AUTH: (&str, &str) = ("authorization", "Bearer metrics-key");

/// How the mock engine answers inference requests.
#[derive(Clone, Debug)]
pub enum Script {
    Json(&'static str),
    /// `text/event-stream`: the chunks, with `gap` before each one after the first.
    Sse {
        chunks: Vec<Bytes>,
        gap: Duration,
    },
    /// A stream that never ends.
    Endless {
        chunk: Bytes,
        gap: Duration,
    },
    /// `text/event-stream`: the chunks at once, then nothing more until the response is dropped, as an engine
    /// still working on its next token sends.
    Quiet {
        chunks: Vec<Bytes>,
    },
    /// JSON, but the response headers come only after this delay.
    Slow(Duration),
    /// A body of `content_type`: the chunks, `gap` apart, then the connection breaks, as a killed engine's does.
    Broken {
        content_type: &'static str,
        chunks: Vec<Bytes>,
        gap: Duration,
    },
}

/// A request the mock engine received.
#[derive(Clone, Debug)]
pub struct Recorded {
    pub path: String,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(|v| v.to_str().unwrap())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

struct MockState {
    script: Script,
    requests: Mutex<Vec<Recorded>>,
    healthy: AtomicBool,
    /// The permits gated answers wait for, granted in the order they began to wait.
    gate: Semaphore,
    dropped: mpsc::UnboundedSender<(Instant, Option<String>)>,
    /// Inference requests the engine is working on (from their arrival to their response's end or drop), and the
    /// most at once.
    active: AtomicUsize,
    peak: AtomicUsize,
}

/// An in-process engine on 127.0.0.1, or on a Unix socket ([`Mock::start_unix`]).
pub struct Mock {
    /// Its TCP address; for a mock on a Unix socket, the unspecified address, which no gateway reaches.
    pub addr: SocketAddr,
    /// The gateway's `upstream` for it: `http://` and its address, or `unix:` and its socket's path.
    pub upstream: String,
    /// The connections it has accepted.
    pub connections: Arc<AtomicUsize>,
    state: Arc<MockState>,
    dropped: mpsc::UnboundedReceiver<(Instant, Option<String>)>,
}

/// Sends the time it is dropped, with the request's tag: held by a response body, or by a handler that has
/// not answered yet. It counts the request as one the engine works on while it lives.
struct DropSignal(
    mpsc::UnboundedSender<(Instant, Option<String>)>,
    Option<String>,
    Arc<MockState>,
);

impl DropSignal {
    fn new(state: &Arc<MockState>, tag: Option<String>) -> Self {
        let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
        state.peak.fetch_max(active, Ordering::SeqCst);
        Self(state.dropped.clone(), tag, Arc::clone(state))
    }
}

impl Drop for DropSignal {
    fn drop(&mut self) {
        self.2.active.fetch_sub(1, Ordering::SeqCst);
        let _ = self.0.send((Instant::now(), self.1.take()));
    }
}

impl Mock {
    pub async fn start(script: Script) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let accepted = Arc::clone(&connections);
        let listener = listener.tap_io(move |tcp| {
            tcp.set_nodelay(true).unwrap();
            accepted.fetch_add(1, Ordering::SeqCst);
        });
        let mut mock = Self::serve(script, listener, Arc::clone(&connections));
        (mock.addr, mock.upstream) = (addr, format!("http://{addr}"));
        mock
    }

    /// A mock engine on 127.0.0.1 whose first connection closes at once, before any answer.
    pub async fn start_flaky(script: Script) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let listener = DropsFirst {
            listener,
            dropped: false,
        };
        let mut mock = Self::serve(script, listener, Arc::new(AtomicUsize::new(0)));
        (mock.addr, mock.upstream) = (addr, format!("http://{addr}"));
        mock
    }

    /// A mock engine on a new Unix socket, as the engine serves on its GPU node or a tunnel on the control host.
    pub async fn start_unix(script: Script) -> Self {
        Self::start_unix_at(socket_path(), script).await
    }

    /// A mock engine on the Unix socket `path`, as an engine that comes back on its socket.
    pub async fn start_unix_at(path: PathBuf, script: Script) -> Self {
        let listener = UnixListener::bind(&path).unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let accepted = Arc::clone(&connections);
        let listener = listener.tap_io(move |_| {
            accepted.fetch_add(1, Ordering::SeqCst);
        });
        let mut mock = Self::serve(script, listener, connections);
        mock.upstream = format!("unix:{}", path.display());
        mock
    }

    fn serve<L>(script: Script, listener: L, connections: Arc<AtomicUsize>) -> Self
    where
        L: axum::serve::Listener,
        L::Addr: std::fmt::Debug,
    {
        let (dropped_tx, dropped) = mpsc::unbounded_channel();
        let state = Arc::new(MockState {
            script,
            requests: Mutex::default(),
            healthy: AtomicBool::new(true),
            gate: Semaphore::new(0),
            dropped: dropped_tx,
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(answer))
            .route("/v1/completions", post(answer))
            .route("/v1/responses", post(answer))
            .route("/v1/models", get(models))
            .route(
                "/metrics",
                get(|| async { ([(CONTENT_TYPE, ENGINE_METRICS_TYPE)], ENGINE_METRICS) }),
            )
            .route(
                "/health",
                get(|State(state): State<Arc<MockState>>| async move {
                    if state.healthy.load(Ordering::SeqCst) {
                        StatusCode::OK
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE
                    }
                }),
            )
            .with_state(Arc::clone(&state));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            addr: SocketAddr::from(([0, 0, 0, 0], 0)),
            upstream: String::new(),
            connections,
            state,
            dropped,
        }
    }

    pub fn set_healthy(&self, healthy: bool) {
        self.state.healthy.store(healthy, Ordering::SeqCst);
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.state.requests.lock().unwrap().clone()
    }

    pub fn last(&self) -> Recorded {
        self.requests()
            .pop()
            .expect("the mock engine received a request")
    }

    /// When the next response body (or pending handler) was dropped.
    pub async fn next_drop(&mut self, within: Duration) -> Instant {
        tokio::time::timeout(within, self.dropped.recv())
            .await
            .expect("the mock engine's response was not dropped in time")
            .unwrap()
            .0
    }

    /// When the response of the request tagged `tag` was dropped; drops of other requests are skipped.
    pub async fn dropped(&mut self, tag: &str, within: Duration) -> Instant {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            let (at, dropped) = tokio::time::timeout_at(deadline, self.dropped.recv())
                .await
                .unwrap_or_else(|_| panic!("the response of {tag} was not dropped in time"))
                .unwrap();
            if dropped.as_deref() == Some(tag) {
                return at;
            }
        }
    }

    /// Lets `n` gated answers go on, the longest waiting first.
    pub fn release(&self, n: usize) {
        self.state.gate.add_permits(n);
    }

    /// The most inference requests the engine worked on at once.
    pub fn peak(&self) -> usize {
        self.state.peak.load(Ordering::SeqCst)
    }

    /// The tags of the requests received, in order of arrival.
    pub fn tags(&self) -> Vec<String> {
        self.requests()
            .iter()
            .filter_map(|r| r.header(TAG).map(str::to_owned))
            .collect()
    }

    /// Waits until the mock engine has received `n` requests in all.
    pub async fn await_requests(&self, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.requests().len() < n {
            assert!(
                Instant::now() < deadline,
                "{} requests reached the mock engine, expected {n}: {:?}",
                self.requests().len(),
                self.tags()
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

impl MockState {
    /// Waits for a permit of the gate.
    async fn pass(&self) {
        self.gate.acquire().await.unwrap().forget();
    }

    async fn record(&self, request: Request) {
        let (parts, body) = request.into_parts();
        let body = body.collect().await.unwrap().to_bytes();
        self.requests.lock().unwrap().push(Recorded {
            path: parts.uri.path().to_owned(),
            headers: parts.headers,
            body,
        });
    }
}

async fn models(State(state): State<Arc<MockState>>, request: Request) -> &'static str {
    state.record(request).await;
    r#"{"object":"list","data":[{"id":"qwen","object":"model"}]}"#
}

async fn answer(State(state): State<Arc<MockState>>, request: Request) -> Response {
    let headers = request.headers();
    let tag = headers.get(TAG).map(|v| v.to_str().unwrap().to_owned());
    let gated = headers.contains_key(GATE);
    let guard = DropSignal::new(&state, tag);
    state.record(request).await;
    match state.script.clone() {
        Script::Json(json) => {
            if gated {
                state.pass().await;
            }
            ([(CONTENT_TYPE, "application/json")], json).into_response()
        }
        Script::Slow(delay) => {
            tokio::time::sleep(delay).await;
            drop(guard);
            ([(CONTENT_TYPE, "application/json")], "{}").into_response()
        }
        Script::Sse { chunks, gap } => {
            let gate = gated.then_some(state);
            let chunks = stream::unfold(
                (chunks.into_iter(), guard, gate, true),
                move |(mut chunks, guard, gate, first)| async move {
                    let chunk = chunks.next()?;
                    if !first {
                        tokio::time::sleep(gap).await;
                    } else if let Some(state) = &gate {
                        state.pass().await;
                    }
                    Some((Ok::<_, Infallible>(chunk), (chunks, guard, gate, false)))
                },
            );
            event_stream(Body::from_stream(chunks))
        }
        Script::Broken {
            content_type,
            chunks,
            gap,
        } => {
            // After the chunks, one error: hyper then closes the connection without ending the body.
            let chunks = stream::unfold(
                (chunks.into_iter(), guard, false),
                move |(mut chunks, guard, broken)| async move {
                    if broken {
                        return None;
                    }
                    tokio::time::sleep(gap).await;
                    let item = chunks
                        .next()
                        .ok_or_else(|| std::io::Error::other("the engine died"));
                    let broken = item.is_err();
                    Some((item, (chunks, guard, broken)))
                },
            );
            ([(CONTENT_TYPE, content_type)], Body::from_stream(chunks)).into_response()
        }
        Script::Quiet { chunks } => {
            let chunks = stream::unfold(
                (chunks.into_iter(), guard),
                |(mut chunks, guard)| async move {
                    match chunks.next() {
                        Some(chunk) => Some((Ok::<_, Infallible>(chunk), (chunks, guard))),
                        None => std::future::pending().await,
                    }
                },
            );
            event_stream(Body::from_stream(chunks))
        }
        Script::Endless { chunk, gap } => {
            let chunks = stream::unfold(guard, move |guard| {
                let chunk = chunk.clone();
                async move {
                    if !gap.is_zero() {
                        tokio::time::sleep(gap).await;
                    }
                    Some((Ok::<_, Infallible>(chunk), guard))
                }
            });
            event_stream(Body::from_stream(chunks))
        }
    }
}

fn event_stream(body: Body) -> Response {
    ([(CONTENT_TYPE, "text/event-stream")], body).into_response()
}

/// A gateway serving TCP 127.0.0.1:0 and a Unix socket whose tenant is `voice`.
pub struct Gateway {
    pub tcp: Target,
    pub unix: Target,
    /// Its metrics, read in process: during a drain, when no listener takes the requests of [`Gateway::metrics`].
    pub exposition: MetricsReader,
    shutdown: Option<oneshot::Sender<()>>,
    done: Option<JoinHandle<()>>,
}

/// Where a test client connects.
#[derive(Clone, Debug)]
pub enum Target {
    Tcp(SocketAddr),
    Unix(PathBuf),
}

/// A Unix socket path unique to this test process and call, short enough for `sun_path`.
/// A listener that closes the first connection it accepts at once, as a connection that breaks on its way to an
/// engine that serves, and hands on every later one.
struct DropsFirst {
    listener: TcpListener,
    dropped: bool,
}

impl axum::serve::Listener for DropsFirst {
    type Io = TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (TcpStream, SocketAddr) {
        loop {
            let accepted = axum::serve::Listener::accept(&mut self.listener).await;
            if std::mem::replace(&mut self.dropped, true) {
                return accepted;
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }
}

/// A socket path of its own, in a directory only this user can enter (mode 700), where the gateway binds sockets; the
/// directory, one for all of the user's test runs, stays.
pub fn socket_path() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        // Short: a socket's path must fit sun_path.
        let user = nix::unistd::Uid::effective();
        let dir = std::env::temp_dir().join(format!("flatshare-gw-sockets-{user}"));
        match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
            Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => {
                panic!("{}: {e}", dir.display())
            }
            _ => std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap(),
        }
        remove_left_by_ended_processes(&dir);
        dir
    });
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    DIR.join(format!("gw-{}-{n}.sock", std::process::id()))
}

/// Removes the files that test processes no longer running left in `dir`, each named for its process
/// (`gw-PID-N.sock`, and the configurations named after them): the sockets the tests bind in systemd's place, which
/// stay after the gateway stops as a socket unit's do, and those of a gateway that did not stop.
fn remove_left_by_ended_processes(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let pid = name
            .to_str()
            .and_then(|name| name.strip_prefix("gw-"))
            .and_then(|rest| rest.split('-').next())
            .and_then(|pid| pid.parse::<i32>().ok());
        let Some(pid) = pid else { continue };
        let ended = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None)
            == Err(nix::errno::Errno::ESRCH);
        if ended {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The metrics key's file of every test gateway: [`METRICS_KEY`], mode 600, written anew by each test process.
pub fn metrics_key_file() -> &'static Path {
    static FILE: LazyLock<PathBuf> = LazyLock::new(|| {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("metrics-key");
        let staged = path.with_extension(std::process::id().to_string());
        std::fs::write(&staged, format!("{METRICS_KEY}\n")).unwrap();
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::rename(&staged, &path).unwrap();
        path
    });
    &FILE
}

/// The cache-salt secret of every test gateway: the bytes 0 to 31, in hex, in a file of mode 600 that each test
/// process writes anew.
pub fn cache_salt_secret_file() -> &'static Path {
    static FILE: LazyLock<PathBuf> = LazyLock::new(|| {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cache-salt-secret");
        let staged = path.with_extension(std::process::id().to_string());
        let secret: String = (0..32u8).map(|b| format!("{b:02x}")).collect();
        std::fs::write(&staged, format!("{secret}\n")).unwrap();
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::rename(&staged, &path).unwrap();
        path
    });
    &FILE
}

pub fn key_digest(key: &str) -> String {
    Sha256::digest(key)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The test configuration: `voice` (trusted, keyless, on the Unix socket), `agents` (trusted, default
/// background, max interactive) and `bench` (untrusted, default interactive). `top` holds extra top-level keys.
pub fn gateway_toml(upstream: SocketAddr, socket: &std::path::Path, top: &str) -> String {
    upstream_toml(&format!("http://{upstream}"), socket, top)
}

/// The test configuration with `upstream` as written: `http://host:port`, or `unix:` and a socket path.
pub fn upstream_toml(upstream: &str, socket: &std::path::Path, top: &str) -> String {
    format!("\nupstream = \"{upstream}\"\n{}", harness_toml(socket, top))
}

/// The test configuration with its TCP listener on `tcp` instead of a port of its own.
pub fn listeners_toml(
    upstream: &str,
    tcp: SocketAddr,
    socket: &std::path::Path,
    top: &str,
) -> String {
    let toml = upstream_toml(upstream, socket, top);
    toml.replace("tcp = \"127.0.0.1:0\"", &format!("tcp = \"{tcp}\""))
}

/// The test configuration's listeners and tenants, after `top`.
fn harness_toml(socket: &std::path::Path, top: &str) -> String {
    format!(
        r#"log_level = "debug"
cache_salt_secret_file = "{secret}"
metrics_key_file = "{metrics_key}"
{top}

[[listen]]
tcp = "127.0.0.1:0"

[[listen]]
unix = "{socket}"
tenant = "voice"

[tenants.voice]
trusted = true
default_class = "realtime"

[tenants.agents]
keys_sha256 = ["{agents}"]
trusted = true
default_class = "background"
max_class = "interactive"

[tenants.bench]
keys_sha256 = ["{bench}"]
default_class = "interactive"
"#,
        socket = socket.display(),
        secret = cache_salt_secret_file().display(),
        metrics_key = metrics_key_file().display(),
        agents = key_digest(AGENTS_KEY),
        bench = key_digest(BENCH_KEY),
    )
}

impl Gateway {
    pub async fn start(upstream: SocketAddr, top: &str) -> Self {
        Self::start_with(upstream, top, Arc::new(LogSink)).await
    }

    pub async fn start_with(upstream: SocketAddr, top: &str, sink: Arc<dyn EventSink>) -> Self {
        let config = Config::from_toml(&gateway_toml(upstream, &socket_path(), top)).unwrap();
        Self::from_config(config, sink).await
    }

    /// A gateway whose `upstream` is `upstream` as written, such as a mock's [`Mock::upstream`].
    pub async fn start_upstream(upstream: &str, top: &str) -> Self {
        let config = Config::from_toml(&upstream_toml(upstream, &socket_path(), top)).unwrap();
        Self::from_config(config, Arc::new(LogSink)).await
    }

    pub async fn from_config(config: Config, sink: Arc<dyn EventSink>) -> Self {
        Self::from_server(Server::bind(config, sink).await.unwrap())
    }

    /// A gateway that takes its listeners from `inherited` where it can, as from systemd's socket activation.
    pub async fn from_inherited(config: Config, inherited: Inherited) -> Self {
        let server = Server::bind_with(config, Arc::new(LogSink), inherited);
        Self::from_server(server.await.unwrap())
    }

    fn from_server(server: Server) -> Self {
        let mut targets = server
            .local_addrs()
            .unwrap()
            .into_iter()
            .map(|addr| match addr {
                ListenAddr::Tcp(addr) => Target::Tcp(addr),
                ListenAddr::Unix(path) => Target::Unix(path),
            });
        let (tcp, unix) = (targets.next().unwrap(), targets.next().unwrap());
        let exposition = server.metrics();
        let (shutdown, stop) = oneshot::channel::<()>();
        let done = tokio::spawn(server.run(async move {
            let _ = stop.await;
        }));
        Self {
            tcp,
            unix,
            exposition,
            shutdown: Some(shutdown),
            done: Some(done),
        }
    }

    /// Starts the shutdown; [`Gateway::stopped`] waits for the drain.
    pub fn shut_down(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }

    pub async fn stopped(&mut self) {
        self.shut_down();
        if let Some(done) = self.done.take() {
            done.await.unwrap();
        }
    }

    /// The Prometheus exposition, from the private socket, which asks for no key.
    pub async fn metrics(&self) -> String {
        let reply = call(&self.unix, "GET", "/metrics", &[], "").await;
        assert_eq!(reply.status, 200);
        String::from_utf8(reply.body.to_vec()).unwrap()
    }

    /// Waits until a series reaches `value`: outcomes are recorded when a relay task ends, just after the
    /// client has seen the end of the response.
    pub async fn await_metric(&self, series: &str, value: f64) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let text = self.metrics().await;
            let current = metric(&text, series);
            if current == Some(value) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "{series} is {current:?}, expected {value}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.shut_down();
    }
}

/// The value of one series in an exposition.
pub fn metric(text: &str, series: &str) -> Option<f64> {
    text.lines()
        .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '))
        .map(|value| value.parse().unwrap())
}

/// A response read in full.
#[derive(Debug)]
pub struct Reply {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(|v| v.to_str().unwrap())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

/// Sends one request on a new connection and reads the whole response.
pub async fn call(
    target: &Target,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> Reply {
    let (response, _connection) = open(target, method, path, headers, body).await;
    let (parts, body) = response.into_parts();
    Reply {
        status: parts.status,
        headers: parts.headers,
        body: body.collect().await.unwrap().to_bytes(),
    }
}

/// Sends a chat completion on a new connection in the background; the handle gives the whole response.
pub fn spawn_call(
    target: &Target,
    headers: Vec<(&'static str, &'static str)>,
    body: &'static str,
) -> JoinHandle<Reply> {
    let target = target.clone();
    tokio::spawn(async move { call(&target, "POST", "/v1/chat/completions", &headers, body).await })
}

/// Sends one request on a new connection and returns the response head with its connection task. Aborting
/// the task closes the connection, as a client that goes away would.
pub async fn open(
    target: &Target,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (http::Response<Incoming>, JoinHandle<()>) {
    let (mut sender, connection) = connect(target).await;
    let response = sender
        .send_request(request(method, path, headers, body))
        .await
        .unwrap();
    (response, connection)
}

pub fn request(
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> http::Request<Full<Bytes>> {
    let mut request = http::Request::builder()
        .method(Method::from_bytes(method.as_bytes()).unwrap())
        .uri(path)
        .header("host", "gateway");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    request
        .body(Full::new(Bytes::copy_from_slice(body.as_bytes())))
        .unwrap()
}

/// A new HTTP/1.1 connection and the task that drives it.
pub async fn connect(target: &Target) -> (http1::SendRequest<Full<Bytes>>, JoinHandle<()>) {
    match target {
        Target::Tcp(addr) => {
            let stream = TcpStream::connect(addr).await.unwrap();
            stream.set_nodelay(true).unwrap();
            handshake(TokioIo::new(stream)).await
        }
        Target::Unix(path) => {
            handshake(TokioIo::new(UnixStream::connect(path).await.unwrap())).await
        }
    }
}

async fn handshake<I>(io: TokioIo<I>) -> (http1::SendRequest<Full<Bytes>>, JoinHandle<()>)
where
    I: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (sender, connection) = http1::handshake(io).await.unwrap();
    (
        sender,
        tokio::spawn(async move {
            let _ = connection.await;
        }),
    )
}

pub const AGENTS_AUTH: (&str, &str) = ("authorization", "Bearer agents-key");
pub const BENCH_AUTH: (&str, &str) = ("authorization", "Bearer bench-key");

/// Cache salts under [`cache_salt_secret_file`]: the first 32 hex digits of
/// `printf <tenant> | openssl dgst -sha256 -mac HMAC -macopt hexkey:000102...1f`.
pub const AGENTS_SALT: &str = "3c112a3140ece6fbba30769936c5f17f";
pub const BENCH_SALT: &str = "770212abeff38f7d5eb73f9c614f45e6";
pub const VOICE_SALT: &str = "b9d705094e95b446146e3c5671bc3c6d";

/// Records accepted lifecycle events.
#[derive(Debug, Default)]
pub struct RecordingSink(pub Mutex<Vec<AcceptedEvent>>);

impl EventSink for RecordingSink {
    fn accept(&self, event: AcceptedEvent) {
        self.0.lock().unwrap().push(event);
    }
}

pub const CHAT: &str = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#;

/// Awaits `future` for a few seconds at most: what a drain or a reload does takes well under a second here, so a
/// test whose gateway does not do it fails instead of waiting for the timeouts it would hit instead.
pub async fn soon<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("done within 3 s")
}

/// A chat completion's stream whose response has begun: its body after the first frame, which `bytes` holds.
pub struct Open {
    body: Incoming,
    bytes: Vec<u8>,
    _connection: JoinHandle<()>,
}

pub async fn begin(target: &Target, headers: &[(&str, &str)]) -> Open {
    let (response, connection) = open(target, "POST", "/v1/chat/completions", headers, CHAT).await;
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    Open {
        body,
        bytes: first.to_vec(),
        _connection: connection,
    }
}

/// Reads a begun stream to its end: its bytes, whether it ended with an error rather than cleanly, and when.
pub fn finish(mut open: Open) -> JoinHandle<(Vec<u8>, bool, Instant)> {
    tokio::spawn(async move {
        loop {
            match open.body.frame().await {
                Some(Ok(frame)) => open.bytes.extend_from_slice(&frame.into_data().unwrap()),
                Some(Err(_)) => return (open.bytes, true, Instant::now()),
                None => return (open.bytes, false, Instant::now()),
            }
        }
    })
}

/// What a raw client got: the answer's head and body as text, and how long after its head was sent.
pub struct Answer {
    pub text: String,
    pub after: Duration,
}

impl Answer {
    pub fn status(&self) -> &str {
        self.text.get(9..12).unwrap_or_default()
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        let head = self.text.split("\r\n\r\n").next()?;
        head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }

    pub fn json(&self) -> serde_json::Value {
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
pub fn raw(
    addr: SocketAddr,
    head: String,
    parts: Vec<(Duration, Vec<u8>)>,
) -> std::thread::JoinHandle<Answer> {
    std::thread::spawn(move || {
        let mut stream = std::net::TcpStream::connect(addr).unwrap();
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
pub fn head(auth: &str, framing: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer {auth}\r\n\
         Content-Type: application/json\r\n{framing}\r\n\r\n"
    )
}

pub fn tcp(gw: &Gateway) -> SocketAddr {
    match gw.tcp {
        Target::Tcp(addr) => addr,
        Target::Unix(_) => unreachable!("the TCP listener"),
    }
}

pub async fn answered(client: std::thread::JoinHandle<Answer>) -> Answer {
    tokio::task::spawn_blocking(move || client.join().unwrap())
        .await
        .unwrap()
}
