//! Listeners, routes and the process lifecycle: bind every listener, serve, and drain on shutdown.

use std::fmt::Display;
use std::future::{Future, IntoFuture};
use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, get, post};
use axum::serve::ListenerExt;
use http::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use http::{Method, StatusCode, Uri};
use lh_proto::{LifecycleEvent, SloClass};
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::sync::watch;
use tokio::task::JoinSet;

use crate::auth::{self, Tenants};
use crate::config::{Config, ListenAddr, MetricsKey, PrioritySignals};
use crate::events::{self, EventSink, Events};
use crate::exchange::Exchange;
use crate::metrics::{Metrics, Route};
use crate::placement::{self, Node, Placement};
use crate::proxy::{self, Upstream};
use crate::systemd::Inherited;

/// `GET /ready` succeeds while the upstream's `/health` has answered within this window.
const READY_WINDOW: Duration = Duration::from_secs(1);

/// How long `GET /metrics/engine` waits for the engine's `/metrics`: Prometheus' default scrape timeout.
const ENGINE_METRICS_TIMEOUT: Duration = Duration::from_secs(10);

/// State shared by every listener.
#[derive(Debug)]
pub(crate) struct Shared {
    pub tenants: Tenants,
    /// The GPU nodes: the one `upstream`, or `[nodes]` by name.
    pub nodes: Vec<Node>,
    /// With `[nodes]`, the sessions placed on them.
    pub placement: Option<Arc<Placement>>,
    pub upstream_priority: PrioritySignals,
    pub metrics: Arc<Metrics>,
    pub events: Events,
    pub max_body_bytes: usize,
    pub stream_buffer_frames: usize,
    pub stall_timeout: Duration,
    pub first_byte_timeout: Duration,
    pub drain: Drain,
    /// What `GET /metrics` asks of a request on a TCP listener.
    metrics_key: Option<MetricsKey>,
    /// When the one upstream's `/health` last succeeded.
    healthy_at: Mutex<Option<Instant>>,
}

impl Shared {
    /// Accepted lifecycle events, in the order they were received, to the admission point of their session's
    /// node.
    pub fn lifecycle(&self, events: &[LifecycleEvent]) {
        match &self.placement {
            None => {
                if let Some(admission) = &self.nodes[0].admission {
                    admission.events(events);
                }
            }
            Some(placement) => placement.events(&self.nodes, events),
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
    /// socket that no listener names is an error: nothing would accept on it.
    pub async fn bind_with(
        config: Config,
        sink: Arc<dyn EventSink>,
        mut inherited: Inherited,
    ) -> io::Result<Self> {
        let tenants = Tenants::new(&config.tenants, &config.cache_salt_secret);
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
        let idle_timeout = config.upstream_idle_timeout;
        let admission = config.admission.as_ref();
        let (metrics, nodes, placement) = match &config.placement {
            None => {
                let metrics = Arc::new(Metrics::new());
                let upstream = Upstream::new(config.upstream, idle_timeout);
                let node = Node::new(None, upstream, admission, Arc::clone(&metrics), [true; 3]);
                (metrics, vec![node], None)
            }
            Some(placement) => {
                let names: Vec<Arc<str>> = placement
                    .nodes
                    .iter()
                    .map(|node| node.name.as_str().into())
                    .collect();
                let metrics = Arc::new(Metrics::with_nodes(&names, placement.policy));
                let nodes: Vec<Node> = placement
                    .nodes
                    .iter()
                    .zip(names)
                    .map(|(node, name)| {
                        let upstream = Upstream::new(node.upstream.clone(), idle_timeout);
                        let own = Arc::new(metrics.on(&name));
                        Node::new(Some(name), upstream, admission, own, node.serves)
                    })
                    .collect();
                let placement = Placement::new(placement, &nodes, Arc::clone(&metrics));
                (metrics, nodes, Some(placement))
            }
        };
        let shared = Shared {
            tenants,
            nodes,
            placement,
            upstream_priority: config.upstream_priority,
            metrics,
            events: Events::new(config.event_dedup_capacity, config.event_dedup_ttl, sink),
            max_body_bytes: config.max_body_bytes,
            stream_buffer_frames: config.stream_buffer_frames,
            stall_timeout: config.stall_timeout,
            first_byte_timeout: config.first_byte_timeout,
            drain: Drain::new(config.background_drain),
            metrics_key: config.metrics_key,
            healthy_at: Mutex::new(None),
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
        let admission = shared.nodes[0].admission.as_ref();
        let admission = admission.map(|admission| admission.policy());
        let listeners = self.local_addrs().unwrap_or_default();
        match &shared.placement {
            None => tracing::info!(
                upstream = %shared.nodes[0].upstream.addr(),
                listeners = ?listeners,
                inherited = self.inherited,
                admission = ?admission,
                "lh-gateway serving"
            ),
            Some(placement) => {
                let nodes: Vec<String> = shared
                    .nodes
                    .iter()
                    .map(|node| {
                        format!(
                            "{}={}",
                            node.name.as_deref().unwrap_or_default(),
                            node.upstream.addr()
                        )
                    })
                    .collect();
                tracing::info!(
                    nodes = ?nodes,
                    placement = placement.policy().as_str(),
                    listeners = ?listeners,
                    inherited = self.inherited,
                    admission = ?admission,
                    "lh-gateway serving"
                );
                for index in 0..shared.nodes.len() {
                    let probe =
                        placement::probe(Arc::downgrade(shared), index, placement.probe_interval());
                    tokio::spawn(probe);
                }
            }
        }
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
    let mut router = Router::new();
    // Only a gateway with [nodes] places sessions: with one upstream, /v1/sessions is no route, as before.
    if app.shared.placement.is_some() {
        router = router.route("/v1/sessions", post(placement::post));
    }
    router
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
/// the engine without a TCP route to it. With `[nodes]`, `?node=NAME` names the engine.
async fn engine_metrics(State(app): State<App>, headers: HeaderMap, uri: Uri) -> Response {
    if let Some(refused) = metrics_refused(&app, &headers) {
        return refused;
    }
    let shared = &app.shared;
    let fail = |status, code, message: String| {
        Exchange::new(&shared.metrics, Route::Other).error(status, code, message)
    };
    let named = uri
        .query()
        .and_then(|query| query.split('&').find_map(|pair| pair.strip_prefix("node=")));
    let node = match (&shared.placement, named) {
        (None, None) => &shared.nodes[0],
        (Some(_), Some(name)) => match shared
            .nodes
            .iter()
            .find(|n| n.name.as_deref() == Some(name))
        {
            Some(node) => node,
            None => {
                return fail(
                    StatusCode::NOT_FOUND,
                    "not_found",
                    format!("no node {name:?}"),
                );
            }
        },
        (Some(_), None) => {
            let message = "the gateway fronts several nodes: name one, /metrics/engine?node=NAME";
            return fail(StatusCode::BAD_REQUEST, "invalid_request", message.into());
        }
        (None, Some(_)) => {
            let message = "the gateway fronts one engine, which has no node name";
            return fail(StatusCode::BAD_REQUEST, "invalid_request", message.into());
        }
    };
    match node.upstream.metrics(ENGINE_METRICS_TIMEOUT).await {
        Ok(response) => response,
        Err(message) => fail(StatusCode::BAD_GATEWAY, "upstream_unavailable", message),
    }
}

/// 200 while the upstream's `/health` has answered within [`READY_WINDOW`]; otherwise it asks once more. 503 at once
/// while a request has found the upstream broken and its probe has not answered since (placement::recover). With
/// `[nodes]`, 200 while a node answers its readiness probe.
async fn ready(State(app): State<App>) -> Response {
    let shared = &app.shared;
    if shared.placement.is_some() {
        return if shared.nodes.iter().any(Node::ready) {
            (StatusCode::OK, "ready\n").into_response()
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "no node is ready\n").into_response()
        };
    }
    if !shared.nodes[0].ready() {
        return (StatusCode::SERVICE_UNAVAILABLE, "upstream not ready\n").into_response();
    }
    let upstream = &shared.nodes[0].upstream;
    let last = *shared
        .healthy_at
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let fresh = last.is_some_and(|at| at.elapsed() < READY_WINDOW);
    if fresh || upstream.healthy(READY_WINDOW).await {
        if !fresh {
            *shared
                .healthy_at
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
        }
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

/// Binds a Unix socket with mode 0600. A socket file nobody accepts on is stale (its process died) and is
/// replaced; a live socket, or a file that is not a socket, is an error.
async fn bind_unix(path: &Path) -> io::Result<(UnixListener, SocketFile)> {
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
}
