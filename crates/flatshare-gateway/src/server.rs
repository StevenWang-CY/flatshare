//! Listeners, routes and the process lifecycle: bind every listener, serve, and drain on shutdown.

use std::fmt::Display;
use std::future::{Future, IntoFuture};
use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, get, post};
use axum::serve::ListenerExt;
use flatshare_proto::{LifecycleEvent, SloClass};
use http::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use http::{Method, StatusCode, Uri};
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::sync::{Notify, watch};
use tokio::task::JoinSet;

use crate::auth::{self, Tenant, Tenants};
use crate::config::{ClassConfig, Config, ListenAddr, MetricsKey, PrioritySignals};
use crate::events::{self, EventSink, Events};
use crate::exchange::Exchange;
use crate::harness::Admission;
use crate::intake::{self, Intake};
use crate::ledger::{self, Followed, Ledger, Load};
use crate::metrics::{Metrics, Route};
use crate::proxy;
use crate::systemd::Inherited;
use crate::upstream::Upstream;

/// `GET /ready` succeeds while the upstream's `/health` has answered within this window.
const READY_WINDOW: Duration = Duration::from_secs(1);

/// How long `GET /metrics/engine` waits for the engine's `/metrics`: Prometheus' default scrape timeout.
const ENGINE_METRICS_TIMEOUT: Duration = Duration::from_secs(10);

/// State shared by every listener.
#[derive(Debug)]
pub(crate) struct Shared {
    pub tenants: Tenants,
    /// The engine.
    pub upstream: Upstream,
    /// The bounds on the requests in flight and their bodies, whatever the admission policy.
    pub intake: Arc<Intake>,
    /// The admission point; `None` under policy "none", where requests go upstream at once.
    pub admission: Option<Arc<Admission>>,
    /// Told of every change to what the intake and the admission point hold, for a successor's ledger.
    pub changed: Arc<Notify>,
    pub upstream_priority: PrioritySignals,
    pub metrics: Arc<Metrics>,
    pub events: Events,
    pub max_body_bytes: usize,
    /// What each class's requests may ask for, by [`SloClass`].
    pub classes: [ClassConfig; 3],
    /// The choices a request may ask for at most.
    pub max_choices: u32,
    pub stream_buffer_frames: usize,
    pub stall_timeout: Duration,
    pub first_byte_timeout: Duration,
    pub drain: Drain,
    /// What `GET /metrics` asks of a request on a TCP listener.
    metrics_key: Option<MetricsKey>,
}

impl Shared {
    /// What the gateway holds on its engine, with what it counts of a predecessor's: a successor's ledger line.
    pub fn load(&self) -> Load {
        Load {
            intake: self.intake.held(),
            admission: self
                .admission
                .as_ref()
                .map(|admission| admission.held().into()),
            closed: self.intake.is_closed(),
        }
    }

    /// What a reload's predecessor holds now, from its ledger, counted with the gateway's own until it exits.
    pub fn set_predecessor(&self, load: &Load) {
        self.intake.set_predecessor(load.intake.clone());
        if let Some(admission) = &self.admission {
            admission.set_predecessor(load.elsewhere());
        }
    }

    /// Admits nothing more: the gateway handed its sockets to a successor. Its waiting requests get 503, and the
    /// requests already admitted run on.
    fn close(&self) {
        self.intake.close();
        if let Some(admission) = &self.admission {
            admission.close();
        }
    }

    /// The configuration of `class`'s requests.
    pub fn class(&self, class: SloClass) -> &ClassConfig {
        &self.classes[class as usize]
    }

    /// Accepted lifecycle events of `tenant`'s sessions, in the order they were received, to the admission point.
    pub fn lifecycle(&self, tenant: &Tenant, events: &[LifecycleEvent]) {
        if let Some(admission) = &self.admission {
            admission.events(tenant, events);
        }
    }
}

/// A drain's cut: the moment, `background_drain_s` into a drain, when the background requests still running are
/// ended. It never comes before a drain.
#[derive(Debug)]
pub(crate) struct Drain {
    cut: watch::Sender<bool>,
    /// `background_drain_s`.
    pub grace: Duration,
}

impl Drain {
    fn new(grace: Duration) -> Self {
        Self {
            cut: watch::channel(false).0,
            grace,
        }
    }

    /// What a request of `class` waits on for the cut, which ends background requests only.
    pub fn cut_for(&self, class: Option<SloClass>) -> Option<watch::Receiver<bool>> {
        (class == Some(SloClass::Background)).then(|| self.cut.subscribe())
    }

    fn cut(&self) {
        self.cut.send_replace(true);
    }
}

/// Completes at the drain's cut; never for a request the cut does not end (`None`).
pub(crate) async fn cut(cut: &mut Option<watch::Receiver<bool>>) {
    match cut {
        Some(cut) => {
            let _ = cut.wait_for(|cut| *cut).await;
        }
        None => std::future::pending().await,
    }
}

/// Why the drain's cut ended a request, which its client reads in the error.
pub(crate) fn drain_message(grace: Duration) -> String {
    format!(
        "the gateway is stopping or reloading: background requests end {}s into its drain (background_drain_s); \
         send the request again",
        grace.as_secs_f64()
    )
}

/// The state of one listener's routes.
#[derive(Clone, Debug)]
pub(crate) struct App {
    pub shared: Arc<Shared>,
    /// The tenant of requests on this listener that carry no known API key.
    pub listener_tenant: Option<usize>,
    /// Whether the listener is a Unix socket, which its file mode keeps from other users of the host.
    pub private: bool,
}

/// A gateway with every listener bound.
#[derive(Debug)]
pub struct Server {
    shared: Arc<Shared>,
    listeners: Vec<(Listener, Option<usize>)>,
    /// How many of them it inherited.
    inherited: usize,
    /// The socket files it bound itself, removed when the server stops unless it handed them over; an inherited
    /// socket's stays.
    sockets: Vec<SocketFile>,
    handover: HandOver,
    drain_timeout: Duration,
}

/// Set once a successor serves the gateway's sockets ([`crate::handover`]), before the drain starts: the socket
/// files the gateway bound itself then stay when it stops.
#[derive(Clone, Debug, Default)]
pub struct HandOver(Arc<AtomicBool>);

impl HandOver {
    pub fn done(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn is_done(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
enum Listener {
    Tcp(TcpListener),
    Unix(UnixListener, PathBuf),
}

impl Server {
    /// Binds every configured listener; nothing is served until [`Server::run`]. Accepted lifecycle events go
    /// to `sink`.
    pub async fn bind(config: Config, sink: Arc<dyn EventSink>) -> io::Result<Self> {
        Self::bind_with(config, sink, Inherited::default()).await
    }

    /// Takes each configured listener from `inherited`, the socket of the same address, and binds the others. A
    /// socket that no listener names is an error: nothing would accept on it. So is an admission configuration that
    /// the core refuses, which only a configuration built in code can hold.
    pub async fn bind_with(
        config: Config,
        sink: Arc<dyn EventSink>,
        mut inherited: Inherited,
    ) -> io::Result<Self> {
        let tenants = Tenants::new(&config.tenants, &config.cache_salt_secret);
        let changed = Arc::new(Notify::new());
        // The harness, and its metric families, only with an admission policy; made before anything is bound, so
        // that a configuration it refuses changes nothing.
        let admission = config
            .admission
            .as_ref()
            .map(|admission| Admission::new(admission, config.max_queued, Arc::clone(&changed)))
            .transpose()
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
        let mut listeners = Vec::with_capacity(config.listeners.len());
        let mut sockets = Vec::new();
        let given = inherited.len();
        // Every inherited socket is matched before anything is bound, so that a start that fails changes nothing.
        let taken: Vec<_> = config
            .listeners
            .iter()
            .map(|listen| match &listen.addr {
                ListenAddr::Tcp(addr) => (inherited.take_tcp(*addr), None),
                ListenAddr::Unix(path) => (None, inherited.take_unix(path)),
            })
            .collect();
        inherited.none_left()?;
        for (listen, taken) in config.listeners.iter().zip(taken) {
            let tenant = listen
                .tenant
                .as_deref()
                .and_then(|name| tenants.index(name));
            let listener = match (&listen.addr, taken) {
                (ListenAddr::Tcp(_), (Some(listener), _)) => {
                    listener.set_nonblocking(true)?;
                    Listener::Tcp(TcpListener::from_std(listener)?)
                }
                (ListenAddr::Tcp(addr), _) => Listener::Tcp(
                    TcpListener::bind(addr)
                        .await
                        .map_err(|e| context(e, addr))?,
                ),
                (ListenAddr::Unix(path), (_, Some(listener))) => {
                    listener.set_nonblocking(true)?;
                    Listener::Unix(UnixListener::from_std(listener)?, path.clone())
                }
                (ListenAddr::Unix(path), _) => {
                    let (listener, file) = bind_unix(path).await?;
                    sockets.push(file);
                    Listener::Unix(listener, path.clone())
                }
            };
            listeners.push((listener, tenant));
        }
        let metrics = Arc::new(match &admission {
            Some(admission) => Metrics::with_harness(admission.metrics()),
            None => Metrics::new(),
        });
        let upstream = Upstream::new(
            config.upstream,
            config.upstream_idle_timeout,
            metrics.node_ready(),
        );
        let limits = intake::Limits {
            in_flight: config.max_in_flight,
            classes: config.classes.each_ref().map(|class| class.max_in_flight),
            body_bytes: config.max_buffered_body_bytes,
        };
        let intake = Intake::new(
            limits,
            &tenants,
            Arc::clone(&changed),
            metrics.intake_gauges(),
        );
        let shared = Shared {
            tenants,
            upstream,
            intake: Arc::new(intake),
            admission,
            changed,
            upstream_priority: config.upstream_priority,
            metrics,
            events: Events::new(config.event_dedup_capacity, config.event_dedup_ttl, sink),
            max_body_bytes: config.max_body_bytes,
            classes: config.classes,
            max_choices: config.max_choices,
            stream_buffer_frames: config.stream_buffer_frames,
            stall_timeout: config.stall_timeout,
            first_byte_timeout: config.first_byte_timeout,
            drain: Drain::new(config.background_drain),
            metrics_key: config.metrics_key,
        };
        Ok(Self {
            shared: Arc::new(shared),
            listeners,
            inherited: given,
            sockets,
            handover: HandOver::default(),
            drain_timeout: config.drain_timeout,
        })
    }

    /// The gateway's metrics, readable in process at any time, before it serves and after it stops: what
    /// `GET /metrics` answers.
    pub fn metrics(&self) -> MetricsReader {
        MetricsReader(Arc::clone(&self.shared.metrics))
    }

    /// Copies of the listening sockets, in configuration order, for a successor to take over.
    pub fn listener_fds(&self) -> io::Result<Vec<OwnedFd>> {
        self.listeners
            .iter()
            .map(|(listener, _)| match listener {
                Listener::Tcp(listener) => listener.as_fd().try_clone_to_owned(),
                Listener::Unix(listener, _) => listener.as_fd().try_clone_to_owned(),
            })
            .collect()
    }

    /// What tells [`Server::run`] that a successor serves the sockets.
    pub fn handover(&self) -> HandOver {
        self.handover.clone()
    }

    /// The gateway's ledger: what it holds on its engine, for a reload's successor to count ([`crate::ledger`]).
    pub fn ledger(&self) -> Ledger {
        Ledger(Arc::downgrade(&self.shared))
    }

    /// Counts what the predecessor whose ledger is at `path` holds with this gateway's own, until it exits. The
    /// returned [`Followed`] completes once the predecessor admits nothing more, which a successor waits for before
    /// it serves.
    pub async fn follow(&self, path: &Path) -> io::Result<Followed> {
        ledger::follow(path, Arc::downgrade(&self.shared)).await
    }

    /// Where the listeners are bound, in configuration order, with the assigned port for TCP port 0.
    pub fn local_addrs(&self) -> io::Result<Vec<ListenAddr>> {
        self.listeners
            .iter()
            .map(|(listener, _)| match listener {
                Listener::Tcp(listener) => listener.local_addr().map(ListenAddr::Tcp),
                Listener::Unix(_, path) => Ok(ListenAddr::Unix(path.clone())),
            })
            .collect()
    }

    /// Serves until `shutdown` completes, then stops accepting and lets in-flight requests finish for up to the
    /// drain timeout, except background ones, which are cut `background_drain_s` into the drain. The socket files
    /// it bound itself are removed before it returns, unless a successor serves them ([`Server::handover`]).
    pub async fn run(self, shutdown: impl Future<Output = ()>) {
        let shared = &self.shared;
        let admission = shared
            .admission
            .as_ref()
            .map(|admission| admission.policy());
        let listeners = self.local_addrs().unwrap_or_default();
        tracing::info!(
            upstream = %shared.upstream.addr(),
            listeners = ?listeners,
            inherited = self.inherited,
            admission = ?admission,
            "flatshare-gateway serving"
        );
        // Dropping `stop` tells every listener to shut down.
        let (stop, stopped) = watch::channel(());
        let mut serving = JoinSet::new();
        for (listener, listener_tenant) in self.listeners {
            let router = router(App {
                shared: Arc::clone(&self.shared),
                listener_tenant,
                private: matches!(listener, Listener::Unix(..)),
            });
            let mut stopped = stopped.clone();
            let signal = async move {
                let _ = stopped.changed().await;
            };
            match listener {
                Listener::Tcp(listener) => {
                    let listener = listener.tap_io(|stream| {
                        if let Err(err) = stream.set_nodelay(true) {
                            tracing::debug!("cannot set TCP_NODELAY: {err}");
                        }
                    });
                    serving.spawn(serve(listener, router, signal));
                }
                Listener::Unix(listener, _) => {
                    serving.spawn(serve(listener, router, signal));
                }
            }
        }
        shutdown.await;
        let grace = shared.drain.grace;
        tracing::info!(
            drain_timeout_s = self.drain_timeout.as_secs_f64(),
            background_drain_s = grace.as_secs_f64(),
            "shutting down: no new connections; draining in-flight requests"
        );
        drop(stop);
        // One admission point owns the admissions: from here on the successor's, which counts what this one holds and
        // serves once this one says that it admits nothing more.
        if self.handover.is_done() {
            shared.close();
            tracing::info!(
                "handed over: admitting nothing more; the successor counts this gateway's requests until it exits"
            );
        }
        let drained = async { while serving.join_next().await.is_some() {} };
        let cut = async {
            tokio::time::sleep(grace).await;
            tracing::info!(
                "background_drain_s reached: cutting the background requests still in flight"
            );
            shared.drain.cut();
            std::future::pending::<()>().await;
        };
        let drain = async {
            tokio::select! {
                () = drained => {}
                () = cut => {}
            }
        };
        if tokio::time::timeout(self.drain_timeout, drain)
            .await
            .is_err()
        {
            tracing::warn!("drain timeout reached; stopping with requests in flight");
        }
        if self.handover.is_done() {
            self.sockets.into_iter().for_each(SocketFile::keep);
        }
    }
}

fn serve<L>(
    listener: L,
    router: Router,
    signal: impl Future<Output = ()> + Send + 'static,
) -> impl Future<Output = io::Result<()>> + Send
where
    L: axum::serve::Listener,
    L::Addr: std::fmt::Debug,
{
    axum::serve(listener, router)
        .with_graceful_shutdown(signal)
        .into_future()
}

fn router(app: App) -> Router {
    Router::new()
        .route("/v1/chat/completions", inference(Route::ChatCompletions))
        .route("/v1/completions", inference(Route::Completions))
        .route("/v1/responses", inference(Route::Responses))
        .route(
            "/v1/models",
            get(|State(app): State<App>, request: Request| proxy::models(app, request)),
        )
        .route("/v1/events", post(events::post))
        .route("/metrics", get(metrics))
        .route("/metrics/engine", get(engine_metrics))
        .route("/health", get(|| async { "ok\n" }))
        .route("/ready", get(ready))
        .fallback(
            |State(app): State<App>, method: Method, uri: Uri| async move {
                Exchange::new(&app.shared.metrics, Route::Other).error(
                    StatusCode::NOT_FOUND,
                    "not_found",
                    format!("no such route: {}", request_line(&method, &uri)),
                )
            },
        )
        .method_not_allowed_fallback(
            |State(app): State<App>, method: Method, uri: Uri| async move {
                Exchange::new(&app.shared.metrics, Route::Other).error(
                    StatusCode::METHOD_NOT_ALLOWED,
                    "method_not_allowed",
                    format!(
                        "the route does not accept this method: {}",
                        request_line(&method, &uri)
                    ),
                )
            },
        )
        .with_state(app)
}

/// The method and path of a request that matched no route, for its error and log line. The `route` label of
/// such a request is "other", so this is the only record of what was asked for; long paths are cut.
fn request_line(method: &Method, uri: &Uri) -> String {
    const MAX_PATH: usize = 256;
    let path = uri.path();
    let end = (0..=MAX_PATH.min(path.len()))
        .rev()
        .find(|&i| path.is_char_boundary(i))
        .unwrap_or(0);
    let cut = if end < path.len() { "…" } else { "" };
    format!("{method} {}{cut}", &path[..end])
}

fn inference(route: Route) -> MethodRouter<App> {
    post(move |State(app): State<App>, request: Request| proxy::inference(app, route, request))
}

/// The 401 of a request that may not read the metrics: one on a TCP listener without the metrics key. On a Unix
/// socket, private to its owner, no key is asked for, as for the requests of its tenant.
fn metrics_refused(app: &App, headers: &HeaderMap) -> Option<Response> {
    let key = app.shared.metrics_key.as_ref();
    let allowed = app.private || key.is_some_and(|key| auth::is_metrics_key(headers, key));
    (!allowed).then(|| {
        Exchange::new(&app.shared.metrics, Route::Other).error(
            StatusCode::UNAUTHORIZED,
            "invalid_api_key",
            "missing or unknown metrics key",
        )
    })
}

async fn metrics(State(app): State<App>, headers: HeaderMap) -> Response {
    if let Some(refused) = metrics_refused(&app, &headers) {
        return refused;
    }
    let content_type =
        HeaderValue::from_static("application/openmetrics-text; version=1.0.0; charset=utf-8");
    ([(CONTENT_TYPE, content_type)], app.shared.metrics.encode()).into_response()
}

/// `GET /metrics/engine`: the engine's own `/metrics`, under the rule of the gateway's, so that a scraper reaches
/// the engine without a TCP route to it.
async fn engine_metrics(State(app): State<App>, headers: HeaderMap) -> Response {
    if let Some(refused) = metrics_refused(&app, &headers) {
        return refused;
    }
    let shared = &app.shared;
    match shared.upstream.metrics(ENGINE_METRICS_TIMEOUT).await {
        Ok(response) => response,
        Err(message) => Exchange::new(&shared.metrics, Route::Other).error(
            StatusCode::BAD_GATEWAY,
            "upstream_unavailable",
            message,
        ),
    }
}

/// 200 while the upstream's `/health` has answered within [`READY_WINDOW`]; otherwise it asks once more. 503 at once
/// while a request has found the upstream broken and its probe has not answered since
/// ([`crate::upstream::recover`]).
async fn ready(State(app): State<App>) -> Response {
    let upstream = &app.shared.upstream;
    if upstream.ready() && upstream.answers(READY_WINDOW).await {
        (StatusCode::OK, "ready\n").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "upstream not ready\n").into_response()
    }
}

/// A gateway's metrics, read in process ([`Server::metrics`]).
#[derive(Clone, Debug)]
pub struct MetricsReader(Arc<Metrics>);

impl MetricsReader {
    /// The OpenMetrics exposition of `GET /metrics`.
    pub fn encode(&self) -> String {
        self.0.encode()
    }
}

/// A Unix socket file the server created, removed when dropped.
#[derive(Debug)]
struct SocketFile(PathBuf);

impl SocketFile {
    /// Leaves the file in place: another process serves it.
    fn keep(self) {
        std::mem::forget(self);
    }
}

impl Drop for SocketFile {
    fn drop(&mut self) {
        if let Err(err) = std::fs::remove_file(&self.0) {
            tracing::warn!("cannot remove {}: {err}", self.0.display());
        }
    }
}

/// Binds a Unix socket with mode 0600, in a directory only the gateway's user can enter. A socket file nobody
/// accepts on is stale (its process died) and is replaced; a live socket, or a file that is not a socket, is an
/// error.
async fn bind_unix(path: &Path) -> io::Result<(UnixListener, SocketFile)> {
    private_directory(path)?;
    if UnixStream::connect(path).await.is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AddrInUse,
            format!(
                "{}: another process is accepting on this socket",
                path.display()
            ),
        ));
    }
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_socket()) {
        std::fs::remove_file(path).map_err(|e| context(e, path.display()))?;
    }
    let listener = UnixListener::bind(path).map_err(|e| context(e, path.display()))?;
    let file = SocketFile(path.to_owned());
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| context(e, path.display()))?;
    Ok((listener, file))
}

/// Refuses a socket `path` in a directory that is not the gateway's user's or that others may enter. bind() creates
/// the socket's file with the process's umask, and only then can its mode be set: meanwhile the directory is what
/// keeps other users from connecting (pathname sockets honor their directory's permissions, unix(7)), as the units'
/// %t/flatshare (mode 700) does. A tenant's socket admits its requests without a key.
fn private_directory(path: &Path) -> io::Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let meta = std::fs::metadata(dir).map_err(|e| context(e, dir.display()))?;
    let user = nix::unistd::Uid::effective().as_raw();
    if meta.uid() == user && meta.mode() & 0o077 == 0 {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "{}: the socket's directory must be the gateway's user's alone (mode 700, as the units' \
             %t/flatshare), so that no other user can connect before the socket's mode is set; {} is mode \
             {:o}, owned by uid {}",
            path.display(),
            dir.display(),
            meta.mode() & 0o7777,
            meta.uid(),
        ),
    ))
}

fn context(err: io::Error, what: impl Display) -> io::Error {
    io::Error::new(err.kind(), format!("{what}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_lines_name_the_method_and_cut_long_paths() {
        let uri: Uri = "/prometheus/metrics?x=1".parse().unwrap();
        assert_eq!(request_line(&Method::GET, &uri), "GET /prometheus/metrics");
        let long: Uri = format!("/{}", "a".repeat(300)).parse().unwrap();
        let line = request_line(&Method::POST, &long);
        assert_eq!(line, format!("POST /{}…", "a".repeat(255)));
    }

    #[tokio::test]
    async fn a_socket_is_bound_only_in_a_directory_that_others_cannot_enter() {
        let dir = std::env::temp_dir().join(format!("flatshare-gw-dir-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("gw.sock");
        let user = nix::unistd::Uid::effective();
        for mode in [0o755, 0o750, 0o711, 0o701] {
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).unwrap();
            let err = bind_unix(&path).await.unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
            let says = format!("{} is mode {mode:o}, owned by uid {user}", dir.display());
            assert!(err.to_string().ends_with(&says), "{err}");
            assert!(!path.exists(), "refused before binding");
        }
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        let (listener, file) = bind_unix(&path).await.unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        drop((listener, file));
        assert!(!path.exists(), "removed with the server's socket file");
        std::fs::remove_dir(&dir).unwrap();
    }
}
