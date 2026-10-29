//! Listeners and the process lifecycle: bind every listener, serve its routes ([`crate::routes`]), and drain on
//! shutdown.

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
use axum::serve::ListenerExt;
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::sync::{Notify, watch};
use tokio::task::JoinSet;

use crate::auth::Tenants;
use crate::canary::{self, Health, Watched};
use crate::config::{Config, Listen, ListenAddr};
use crate::drain::Drain;
use crate::engine::Engine;
use crate::events::{EventSink, Events};
use crate::harness::Admission;
use crate::intake::{self, Intake};
use crate::ledger::{self, Followed, Ledger};
use crate::metrics::Metrics;
use crate::nodes::{self, Nodes};
use crate::routes;
use crate::state::{App, Engines, Shared};
use crate::systemd::Inherited;
use crate::upstream::Upstream;

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
        inherited: Inherited,
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
        let given = inherited.len();
        let (listeners, sockets) = bind_listeners(&config.listeners, &tenants, inherited).await?;
        let drain_timeout = config.drain_timeout;
        let shared = shared(config, tenants, admission, changed, sink);
        Ok(Self {
            shared: Arc::new(shared),
            listeners,
            inherited: given,
            sockets,
            handover: HandOver::default(),
            drain_timeout,
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
        match &shared.engines {
            Engines::One(engine) => tracing::info!(
                upstream = %engine.upstream.addr(),
                listeners = ?listeners,
                inherited = self.inherited,
                admission = ?admission,
                "flatshare-gateway serving"
            ),
            Engines::Nodes(nodes) => {
                let names: Vec<&str> = nodes.list.iter().map(|node| &*node.name).collect();
                tracing::info!(
                    nodes = ?names,
                    listeners = ?listeners,
                    inherited = self.inherited,
                    "flatshare-gateway serving"
                );
            }
        }
        // Dropping `stop` tells every listener to shut down.
        let (stop, stopped) = watch::channel(());
        let mut serving = JoinSet::new();
        for (listener, listener_tenant) in self.listeners {
            let router = routes::router(App {
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
        match &shared.engines {
            Engines::One(engine) => engine.health.drain(),
            Engines::Nodes(nodes) => nodes.drain(),
        }
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

/// Every configured listener, in configuration order, with the tenant it names, and the socket files the gateway
/// bound itself: each listener taken from `inherited`, the socket of the same address, or else bound. Every inherited
/// socket is matched before anything is bound, so that a start that fails changes nothing.
async fn bind_listeners(
    configured: &[Listen],
    tenants: &Tenants,
    mut inherited: Inherited,
) -> io::Result<(Vec<(Listener, Option<usize>)>, Vec<SocketFile>)> {
    let mut listeners = Vec::with_capacity(configured.len());
    let mut sockets = Vec::new();
    let taken: Vec<_> = configured
        .iter()
        .map(|listen| match &listen.addr {
            ListenAddr::Tcp(addr) => (inherited.take_tcp(*addr), None),
            ListenAddr::Unix(path) => (None, inherited.take_unix(path)),
        })
        .collect();
    inherited.none_left()?;
    for (listen, taken) in configured.iter().zip(taken) {
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
    Ok((listeners, sockets))
}

/// The gateway's metrics, with the harness's families under an admission policy and the nodes' with `[[nodes]]`, and
/// what it sends its requests to: the nodes, each with its engine, or the one upstream's engine and its canary.
fn engines(config: &Config, admission: Option<&Arc<Admission>>) -> (Arc<Metrics>, Engines) {
    let names: Vec<Arc<str>> = config
        .nodes
        .iter()
        .map(|n| n.name.as_str().into())
        .collect();
    let node_metrics = (!names.is_empty()).then(|| nodes::metrics::Metrics::new(&names));
    let metrics = Arc::new(match (admission, &node_metrics) {
        (Some(admission), _) => Metrics::with_harness(admission.metrics()),
        (None, Some(nodes)) => Metrics::with_nodes(nodes),
        (None, None) => Metrics::new(),
    });
    let engines = match &node_metrics {
        Some(node_metrics) => Engines::Nodes(Nodes::start(
            &config.nodes,
            config.placement,
            config.canary,
            config.upstream_priority,
            config.upstream_idle_timeout,
            node_metrics,
        )),
        None => {
            let health = Health::new(
                config.canary,
                Watched {
                    gauges: metrics.engine_state(),
                    probes: metrics.probes(),
                    changed: None,
                    node: None,
                },
            );
            let engine = Arc::new(Engine {
                upstream: Upstream::new(
                    config
                        .upstream
                        .clone()
                        .expect("a configuration without [[nodes]] has its upstream"),
                    config.upstream_idle_timeout,
                    metrics.node_ready(),
                ),
                health: Arc::new(health),
                signals: config.upstream_priority,
                changed: Arc::new(Notify::new()),
            });
            if config.canary.is_some() {
                tokio::spawn(canary::run(Arc::downgrade(&engine)));
            }
            Engines::One(engine)
        }
    };
    (metrics, engines)
}

/// The state every listener shares: the tenants, the engines and their metrics, the intake's bounds, the admission
/// point under an admission policy, the lifecycle events' window and `sink`, and the configuration's limits.
fn shared(
    config: Config,
    tenants: Tenants,
    admission: Option<Arc<Admission>>,
    changed: Arc<Notify>,
    sink: Arc<dyn EventSink>,
) -> Shared {
    let (metrics, engines) = engines(&config, admission.as_ref());
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
    Shared {
        tenants,
        engines,
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
        drain: Drain::new(config.background_drain),
        metrics_key: config.metrics_key,
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
