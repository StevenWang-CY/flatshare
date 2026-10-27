//! The engine, as the gateway reaches it: one pooled HTTP/1.1 client over TCP or the engine's Unix socket, and
//! whether its requests go to it.
//!
//! The engine starts ready. A request that finds it broken starts a probe of its `/health` ([`recover`]): at once,
//! then every [`RECOVERY_PROBE`] until it answers. While the probe fails the engine is not ready, and its requests
//! are answered 503 without going upstream; the probe that answers makes it ready again. `GET /ready` asks the same
//! `/health`, at most once a window ([`Upstream::answers`]).

use std::convert::Infallible;
use std::future::Future;
use std::io;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::response::Response;
use bytes::Bytes;
use http::header::CONTENT_TYPE;
use http::uri::{self, Authority, PathAndQuery, Scheme, Uri};
use http_body::{Frame, SizeHint};
use http_body_util::{BodyExt, Limited};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::{Client, ResponseFuture};
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use tokio::net::UnixStream;
use tokio::time::MissedTickBehavior;

use crate::config::UpstreamAddr;
use crate::server::Shared;

/// How often the engine is probed while a request has found it broken, and the `Retry-After` of the requests
/// answered meanwhile.
pub(crate) const RECOVERY_PROBE: Duration = Duration::from_secs(1);

/// The engine: its pooled HTTP/1.1 client, over TCP or its Unix socket, and whether its requests go to it.
#[derive(Debug)]
pub(crate) struct Upstream {
    pool: Pool,
    addr: UpstreamAddr,
    /// The authority of the requests' URIs: the engine's `host:port`, or `localhost` on a socket.
    authority: Authority,
    /// Whether its requests go to it. It starts ready; a request that finds it broken starts its probe
    /// ([`recover`]), and until its `/health` answers, its requests are answered 503 without going upstream.
    ready: AtomicBool,
    /// Whether [`recover`] runs.
    recovering: AtomicBool,
    /// `flatshare_gateway_node_ready`: 1 while it is ready.
    ready_gauge: Gauge,
    /// When its `/health` last answered `GET /ready`.
    healthy_at: Mutex<Option<Instant>>,
}

/// One connection pool, keyed by the requests' authority; a socket's client connects every request to it.
#[derive(Debug)]
enum Pool {
    Tcp(Client<HttpConnector, UpstreamBody>),
    Unix(Client<UnixConnector, UpstreamBody>),
}

impl Pool {
    fn request(&self, request: http::Request<UpstreamBody>) -> ResponseFuture {
        match self {
            Self::Tcp(client) => client.request(request),
            Self::Unix(client) => client.request(request),
        }
    }
}

/// Connects to one Unix socket whatever the URI: the engine's, or a tunnel's on this host.
#[derive(Clone, Debug)]
struct UnixConnector(Arc<Path>);

impl tower_service::Service<Uri> for UnixConnector {
    type Response = TokioIo<UnixStream>;
    type Error = io::Error;
    type Future = Pin<Box<dyn Future<Output = io::Result<TokioIo<UnixStream>>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: Uri) -> Self::Future {
        let path = Arc::clone(&self.0);
        Box::pin(async move { UnixStream::connect(&*path).await.map(TokioIo::new) })
    }
}

impl Upstream {
    /// A client that keeps idle connections for `idle_timeout`. The engine closes idle connections after its
    /// keep-alive, and a request sent on a connection it is closing fails, so the gateway must let go first.
    /// `ready_gauge` says whether its requests go to it.
    pub fn new(addr: UpstreamAddr, idle_timeout: Duration, ready_gauge: Gauge) -> Self {
        let builder = || {
            let mut builder = Client::builder(TokioExecutor::new());
            builder
                .pool_timer(TokioTimer::new())
                .pool_idle_timeout(idle_timeout);
            builder
        };
        let (pool, authority) = match &addr {
            UpstreamAddr::Tcp(authority) => {
                let mut connector = HttpConnector::new();
                connector.set_nodelay(true);
                (Pool::Tcp(builder().build(connector)), authority.clone())
            }
            UpstreamAddr::Unix(path) => {
                let connector = UnixConnector(path.as_path().into());
                let localhost = Authority::from_static("localhost");
                (Pool::Unix(builder().build(connector)), localhost)
            }
        };
        ready_gauge.set(1);
        Self {
            pool,
            addr,
            authority,
            ready: AtomicBool::new(true),
            recovering: AtomicBool::new(false),
            ready_gauge,
            healthy_at: Mutex::new(None),
        }
    }

    pub fn addr(&self) -> &UpstreamAddr {
        &self.addr
    }

    /// Whether its requests go to it.
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// Records whether its `/health` answers, logging a change.
    fn set_ready(&self, ready: bool) {
        self.ready_gauge.set(i64::from(ready));
        if self.ready.swap(ready, Ordering::Relaxed) == ready {
            return;
        }
        if ready {
            tracing::info!("the upstream is ready again: its /health answers");
        } else {
            tracing::warn!(
                "the upstream is not ready: a request found it broken and its /health does not answer; requests \
                 are answered 503 until it does"
            );
        }
    }

    /// Whether the caller, whose request found the engine broken, is to start its probe ([`recover`]): false while
    /// one runs.
    pub fn start_recovery(&self) -> bool {
        !self.recovering.swap(true, Ordering::AcqRel)
    }

    /// The URI of `path_and_query` on the engine.
    pub fn uri(&self, path_and_query: PathAndQuery) -> Uri {
        let mut parts = uri::Parts::default();
        parts.scheme = Some(Scheme::HTTP);
        parts.authority = Some(self.authority.clone());
        parts.path_and_query = Some(path_and_query);
        Uri::from_parts(parts).expect("a scheme, an authority and a path form a valid URI")
    }

    /// Sends a request on a pooled connection; the future completes with the response headers.
    pub fn request(&self, request: http::Request<UpstreamBody>) -> ResponseFuture {
        self.pool.request(request)
    }

    /// The engine's own `GET /metrics`, read in full within `timeout`, with its status and content type.
    pub async fn metrics(&self, timeout: Duration) -> Result<Response, String> {
        // vLLM's exposition with its KV cache histograms is a few hundred kilobytes.
        const LIMIT: usize = 16 << 20;
        let uri = self.uri(PathAndQuery::from_static("/metrics"));
        let request = http::Request::get(uri)
            .body(UpstreamBody::empty())
            .expect("a GET with a valid URI is a valid request");
        let read = async {
            let response = self
                .request(request)
                .await
                .map_err(|err| format!("the engine's /metrics failed: {}", causes(&err)))?;
            let (parts, body) = response.into_parts();
            let body = Limited::new(body, LIMIT)
                .collect()
                .await
                .map_err(|err| format!("cannot read the engine's /metrics: {err}"))?;
            Ok::<_, String>((parts, body.to_bytes()))
        };
        let (parts, body) = tokio::time::timeout(timeout, read)
            .await
            .map_err(|_| format!("the engine's /metrics did not answer within {timeout:?}"))??;
        let mut response = Response::new(Body::from(body));
        *response.status_mut() = parts.status;
        if let Some(content_type) = parts.headers.get(CONTENT_TYPE) {
            response
                .headers_mut()
                .insert(CONTENT_TYPE, content_type.clone());
        }
        Ok(response)
    }

    /// Whether the engine's `/health` answers with success within `timeout`.
    pub async fn healthy(&self, timeout: Duration) -> bool {
        let uri = self.uri(PathAndQuery::from_static("/health"));
        let request = http::Request::get(uri)
            .body(UpstreamBody::empty())
            .expect("a GET with a valid URI is a valid request");
        matches!(
            tokio::time::timeout(timeout, self.request(request)).await,
            Ok(Ok(response)) if response.status().is_success()
        )
    }

    /// Whether the engine's `/health` answered within the last `window`, asking it once more, for at most `window`,
    /// when it did not: `GET /ready`'s answer while the engine is ready.
    pub async fn answers(&self, window: Duration) -> bool {
        let last = *self
            .healthy_at
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if last.is_some_and(|at| at.elapsed() < window) {
            return true;
        }
        if !self.healthy(window).await {
            return false;
        }
        *self
            .healthy_at
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
        true
    }
}

/// Probes the engine's `/health` after a request found it broken: at once, and every [`RECOVERY_PROBE`] until it
/// answers. A failed probe makes it not ready, so that its requests are answered 503 without going upstream, and the
/// probe that answers makes it ready again and ends the task. One request that fails on an engine that serves (a
/// connection that broke) makes no request wait: the first probe answers.
pub(crate) async fn recover(shared: Weak<Shared>) {
    let mut ticks = tokio::time::interval(RECOVERY_PROBE);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        ticks.tick().await;
        let Some(shared) = shared.upgrade() else {
            return;
        };
        let upstream = &shared.upstream;
        if upstream.healthy(RECOVERY_PROBE).await {
            upstream.recovering.store(false, Ordering::Release);
            upstream.set_ready(true);
            return;
        }
        upstream.set_ready(false);
    }
}

/// An error and its causes, outermost first: `client error (Connect): Connection refused (os error 111)`.
pub(crate) fn causes(err: &(dyn std::error::Error + 'static)) -> String {
    let mut message = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        message.push_str(&format!(": {cause}"));
        source = cause.source();
    }
    message
}

/// The body of an upstream request. Its first poll happens when hyper writes the request to a pooled
/// connection, which is where the gateway's overhead for the request ends; it started when the client's body
/// was complete.
#[derive(Debug)]
pub(crate) struct UpstreamBody {
    data: Option<Bytes>,
    sent: Option<(Instant, Histogram)>,
}

impl UpstreamBody {
    /// The body `data`, whose first poll observes in `overhead` the time since `start`.
    pub fn new(data: Bytes, start: Instant, overhead: Histogram) -> Self {
        Self {
            data: Some(data),
            sent: Some((start, overhead)),
        }
    }

    pub fn empty() -> Self {
        Self {
            data: None,
            sent: None,
        }
    }

    /// The body `data`, which times nothing: the gateway's own requests', the canary's.
    pub fn plain(data: Bytes) -> Self {
        Self {
            data: Some(data),
            sent: None,
        }
    }

    /// The same bytes, to send again after a preemption dropped the attempt: the first attempt observed the
    /// overhead.
    pub fn again(&self) -> Self {
        Self {
            data: self.data.clone(),
            sent: None,
        }
    }
}

impl http_body::Body for UpstreamBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        if let Some((received, overhead)) = self.sent.take() {
            overhead.observe(received.elapsed().as_secs_f64());
        }
        Poll::Ready(self.data.take().map(|data| Ok(Frame::data(data))))
    }

    fn is_end_stream(&self) -> bool {
        self.data.is_none()
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::with_exact(self.data.as_ref().map_or(0, |data| data.len() as u64))
    }
}
