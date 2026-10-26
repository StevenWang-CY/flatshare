//! The gateway's Prometheus metrics, in a registry it owns (no global state), rendered at `GET /metrics`, and, with an
//! admission policy, the harness's after them (`harness`).
//!
//! Every label takes its values from a fixed set or from the configuration (tenant names), so the number of series
//! is bounded. Every series except those of requests (labeled with a status) and lifecycle events is created at
//! start, so it exports zero before traffic.

use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;

use flatshare_proto::SloClass;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::{Family, MetricConstructor};
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::{Registry, Unit};

use crate::harness;
use crate::usage::Usage;

/// The API routes, as the `route` label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    ChatCompletions,
    Completions,
    Responses,
    Models,
    Events,
    /// Any path or method the gateway does not serve.
    Other,
}

impl Route {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
            Self::Responses => "/v1/responses",
            Self::Models => "/v1/models",
            Self::Events => "/v1/events",
            Self::Other => "other",
        }
    }
}

/// How a proxied request ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The upstream response was relayed in full (whatever its status).
    Completed,
    /// The client went away first; the gateway closed the upstream request.
    Cancelled,
    /// The client stopped reading for longer than the stall timeout; the gateway aborted both sides.
    Stalled,
    /// The upstream failed: no connection, no response headers in time, or a broken body.
    Error,
    /// The admission policy shed it to make room for a stricter class; the gateway cancelled the upstream
    /// request.
    Shed,
    /// A background request still running `background_drain_s` into a drain: the gateway cut it, with a
    /// `draining` error event on a stream or a 503 before the response.
    Drained,
}

impl Outcome {
    const ALL: [Self; 6] = [
        Self::Completed,
        Self::Cancelled,
        Self::Stalled,
        Self::Error,
        Self::Shed,
        Self::Drained,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Stalled => "stalled",
            Self::Error => "error",
            Self::Shed => "shed",
            Self::Drained => "drained",
        }
    }
}

/// Why the gateway ended an event stream with an error event: its `type`, and the HTTP status as its `code`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorEvent {
    /// The upstream broke the stream off (502).
    Upstream,
    /// A draining gateway cut a background stream (503): the client may send it again.
    Draining,
}

impl ErrorEvent {
    const ALL: [Self; 2] = [Self::Upstream, Self::Draining];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Upstream => "upstream_unavailable",
            Self::Draining => "draining",
        }
    }

    pub const fn code(self) -> u16 {
        match self {
            Self::Upstream => 502,
            Self::Draining => 503,
        }
    }
}

/// Why an upstream request failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UpstreamError {
    /// No connection could be made (502).
    Connect,
    /// The connection failed before the response headers arrived (502).
    Request,
    /// No response headers within the first-byte timeout (504).
    Timeout,
    /// The response body broke off.
    Body,
}

impl UpstreamError {
    const ALL: [Self; 4] = [Self::Connect, Self::Request, Self::Timeout, Self::Body];

    const fn as_str(self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::Request => "request",
            Self::Timeout => "timeout",
            Self::Body => "body",
        }
    }
}

/// Buckets of the gateway's own overhead, from 50 us. 2 ms, the SLO, is a bucket bound.
const OVERHEAD_BUCKETS: [f64; 12] = [
    50e-6, 100e-6, 200e-6, 300e-6, 500e-6, 750e-6, 1e-3, 1.5e-3, 2e-3, 5e-3, 10e-3, 50e-3,
];

/// Buckets of time to first byte and total duration: from realtime turns to long agent turns.
pub(crate) const LATENCY_BUCKETS: [f64; 16] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RequestLabels {
    /// Empty when the request was not authenticated.
    tenant: Option<Arc<str>>,
    route: &'static str,
    /// Empty for routes without a class.
    class: Option<&'static str>,
    status: u16,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct OutcomeLabels {
    class: &'static str,
    outcome: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RouteLabels {
    route: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ClassLabels {
    class: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct FieldLabels {
    field: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct KindLabels {
    kind: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ErrorEventLabels {
    class: &'static str,
    r#type: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct EventLabels {
    r#type: &'static str,
    source: &'static str,
    duplicate: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct UsageLabels {
    class: &'static str,
    kind: &'static str,
}

pub(crate) type Histograms<L> = Family<L, Histogram, fn() -> Histogram>;

/// The gateway's metrics: cloning a handle shares every series.
#[derive(Clone, Debug)]
pub(crate) struct Metrics {
    registry: Arc<Registry>,
    requests: Family<RequestLabels, Counter>,
    outcomes: Family<OutcomeLabels, Counter>,
    overhead: Histograms<RouteLabels>,
    request_body: Histograms<RouteLabels>,
    first_byte: Histograms<ClassLabels>,
    duration: Histograms<ClassLabels>,
    streams_active: Family<ClassLabels, Gauge>,
    sanitized: Family<FieldLabels, Counter>,
    class_clamped: Counter,
    upstream_errors: Family<KindLabels, Counter>,
    error_events: Family<ErrorEventLabels, Counter>,
    events: Family<EventLabels, Counter>,
    usage_tokens: Family<UsageLabels, Counter>,
    /// Whether the engine is ready.
    node_ready: Gauge,
}

impl Metrics {
    /// The metrics of a gateway without an admission policy.
    pub fn new() -> Self {
        Self::build(None)
    }

    /// The metrics of a gateway with an admission point, whose series `harness` holds.
    pub fn with_harness(harness: &harness::Metrics) -> Self {
        Self::build(Some(harness))
    }

    fn build(harness: Option<&harness::Metrics>) -> Self {
        let metrics = Self {
            registry: Arc::new(Registry::default()),
            requests: Family::default(),
            outcomes: Family::default(),
            overhead: Family::new_with_constructor(|| Histogram::new(OVERHEAD_BUCKETS)),
            request_body: Family::new_with_constructor(|| Histogram::new(OVERHEAD_BUCKETS)),
            first_byte: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            duration: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            streams_active: Family::default(),
            sanitized: Family::default(),
            class_clamped: Counter::default(),
            upstream_errors: Family::default(),
            error_events: Family::default(),
            events: Family::default(),
            usage_tokens: Family::default(),
            node_ready: Gauge::default(),
        };
        for class in SloClass::ALL.map(SloClass::as_str) {
            for outcome in Outcome::ALL.map(Outcome::as_str) {
                create(&metrics.outcomes, OutcomeLabels { class, outcome });
            }
            for kind in ["prompt", "completion", "cached"] {
                create(&metrics.usage_tokens, UsageLabels { class, kind });
            }
            create(&metrics.first_byte, ClassLabels { class });
            create(&metrics.duration, ClassLabels { class });
            create(&metrics.streams_active, ClassLabels { class });
            for r#type in ErrorEvent::ALL.map(ErrorEvent::as_str) {
                create(&metrics.error_events, ErrorEventLabels { class, r#type });
            }
        }
        for route in [Route::ChatCompletions, Route::Completions, Route::Responses] {
            let route = route.as_str();
            create(&metrics.overhead, RouteLabels { route });
            create(&metrics.request_body, RouteLabels { route });
        }
        for kind in UpstreamError::ALL.map(UpstreamError::as_str) {
            create(&metrics.upstream_errors, KindLabels { kind });
        }
        for field in crate::sanitize::sanitized_fields() {
            create(&metrics.sanitized, FieldLabels { field });
        }
        metrics.register(harness)
    }

    /// Registers every series, in the order of the exposition: the harness's after the usage tokens.
    fn register(mut self, harness: Option<&harness::Metrics>) -> Self {
        let mut registry = Registry::with_prefix("flatshare_gateway");
        let r = &mut registry;
        r.register(
            "requests",
            "API requests by tenant, route, class and response status (499: the client left before the response)",
            self.requests.clone(),
        );
        r.register(
            "request_outcomes",
            "Proxied requests by class and how they ended",
            self.outcomes.clone(),
        );
        r.register_with_unit(
            "overhead",
            "Time from the end of a request's body to writing the request to the upstream: parse, sanitize, \
             resolve, connection checkout",
            Unit::Seconds,
            self.overhead.clone(),
        );
        r.register_with_unit(
            "request_body",
            "Time from receiving a request's headers to the end of its body: the client's upload, which a \
             direct call to the engine waits for too",
            Unit::Seconds,
            self.request_body.clone(),
        );
        r.register_with_unit(
            "first_byte",
            "Time from receiving a request to the first response body byte from the upstream",
            Unit::Seconds,
            self.first_byte.clone(),
        );
        r.register_with_unit(
            "duration",
            "Time from receiving a request to the end of the upstream response",
            Unit::Seconds,
            self.duration.clone(),
        );
        r.register(
            "streams_active",
            "Responses being relayed from the upstream",
            self.streams_active.clone(),
        );
        r.register(
            "sanitized",
            "Request headers and body fields removed because clients may not set them",
            self.sanitized.clone(),
        );
        r.register(
            "class_clamped",
            "Requests whose class was stricter than their tenant's max_class",
            self.class_clamped.clone(),
        );
        r.register(
            "upstream_errors",
            "Upstream failures by kind",
            self.upstream_errors.clone(),
        );
        r.register(
            "error_events",
            "Error events sent on event streams before the gateway aborted them, by type: upstream_unavailable when \
             the upstream broke the stream off, draining when a draining gateway cut a background stream",
            self.error_events.clone(),
        );
        r.register(
            "events",
            "Lifecycle events by type, source and whether they repeated an earlier event",
            self.events.clone(),
        );
        r.register(
            "usage_tokens",
            "Tokens reported in response usage, by class and kind",
            self.usage_tokens.clone(),
        );
        if let Some(harness) = harness {
            harness.register(r);
        }
        r.register(
            "node_ready",
            "Whether the engine is ready (1), or a request found it broken and its /health has not answered since \
             (0), while its requests are answered 503",
            self.node_ready.clone(),
        );
        self.registry = Arc::new(registry);
        self
    }

    /// The OpenMetrics text exposition.
    pub fn encode(&self) -> String {
        let mut text = String::with_capacity(16 << 10);
        encode(&mut text, &self.registry).expect("writing to a String cannot fail");
        text
    }

    pub fn request(
        &self,
        tenant: Option<&Arc<str>>,
        route: Route,
        class: Option<SloClass>,
        status: u16,
    ) {
        let labels = RequestLabels {
            tenant: tenant.cloned(),
            route: route.as_str(),
            class: class.map(SloClass::as_str),
            status,
        };
        self.requests.get_or_create(&labels).inc();
    }

    pub fn outcome(&self, class: SloClass, outcome: Outcome) {
        let labels = OutcomeLabels {
            class: class.as_str(),
            outcome: outcome.as_str(),
        };
        self.outcomes.get_or_create(&labels).inc();
    }

    /// How long a request's body took to arrive after its headers.
    pub fn request_body(&self, route: Route, elapsed: Duration) {
        let labels = RouteLabels {
            route: route.as_str(),
        };
        self.request_body
            .get_or_create(&labels)
            .observe(elapsed.as_secs_f64());
    }

    /// The overhead histogram of a route, observed when the upstream request is written.
    pub fn overhead(&self, route: Route) -> Histogram {
        self.overhead.get_or_create_owned(&RouteLabels {
            route: route.as_str(),
        })
    }

    pub fn first_byte(&self, class: SloClass, elapsed: Duration) {
        let labels = ClassLabels {
            class: class.as_str(),
        };
        self.first_byte
            .get_or_create(&labels)
            .observe(elapsed.as_secs_f64());
    }

    pub fn duration(&self, class: SloClass, elapsed: Duration) {
        let labels = ClassLabels {
            class: class.as_str(),
        };
        self.duration
            .get_or_create(&labels)
            .observe(elapsed.as_secs_f64());
    }

    /// Counts a stream as active until the returned guard drops.
    pub fn stream_started(&self, class: SloClass) -> GaugeGuard {
        GaugeGuard::new(self.streams_active.get_or_create_owned(&ClassLabels {
            class: class.as_str(),
        }))
    }

    /// The gauge of whether the engine is ready.
    pub fn node_ready(&self) -> Gauge {
        self.node_ready.clone()
    }

    pub fn sanitized(&self, field: &'static str) {
        self.sanitized.get_or_create(&FieldLabels { field }).inc();
    }

    pub fn class_clamped(&self) {
        self.class_clamped.inc();
    }

    pub fn upstream_error(&self, kind: UpstreamError) {
        let labels = KindLabels {
            kind: kind.as_str(),
        };
        self.upstream_errors.get_or_create(&labels).inc();
    }

    /// An event stream got an error event before its abort.
    pub fn error_event(&self, class: SloClass, event: ErrorEvent) {
        let labels = ErrorEventLabels {
            class: class.as_str(),
            r#type: event.as_str(),
        };
        self.error_events.get_or_create(&labels).inc();
    }

    pub fn event(&self, r#type: &'static str, source: &'static str, duplicate: bool) {
        let labels = EventLabels {
            r#type,
            source,
            duplicate,
        };
        self.events.get_or_create(&labels).inc();
    }

    pub fn usage(&self, class: SloClass, usage: &Usage) {
        let class = class.as_str();
        for (kind, tokens) in [
            ("prompt", usage.prompt),
            ("completion", usage.completion),
            ("cached", usage.cached),
        ] {
            self.usage_tokens
                .get_or_create(&UsageLabels { class, kind })
                .inc_by(tokens);
        }
    }

    #[cfg(test)]
    pub fn sanitized_count(&self, field: &'static str) -> u64 {
        self.sanitized
            .get(&FieldLabels { field })
            .map_or(0, |counter| counter.get())
    }
}

/// Creates a series at zero, so it is exported before its first event.
pub(crate) fn create<S, M, C>(family: &Family<S, M, C>, labels: S)
where
    S: Clone + Hash + Eq,
    C: MetricConstructor<M>,
{
    drop(family.get_or_create(&labels));
}

/// Counts one in a gauge while it lives: an active stream, or a waiting request.
#[derive(Debug)]
pub(crate) struct GaugeGuard(Gauge);

impl GaugeGuard {
    pub(crate) fn new(gauge: Gauge) -> Self {
        gauge.inc();
        Self(gauge)
    }
}

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        self.0.dec();
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// An exposition with each family's series in the order of their labels, so that it reads the same in every
    /// process: a family keeps its series in a hash map. A series' own lines, such as a histogram's sum, count and
    /// buckets, keep their order.
    fn by_labels(exposition: &str) -> String {
        fn flush(out: &mut String, series: &mut Vec<(String, Vec<&str>)>) {
            series.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, lines) in series.drain(..) {
                for line in lines {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
        // A sample's labels without its bucket's `le`: the same for every line of one series.
        fn labels(sample: &str) -> String {
            let Some((_, rest)) = sample.split_once('{') else {
                return String::new();
            };
            let set = rest.split_once('}').map_or(rest, |(set, _)| set);
            set.split(',')
                .filter(|label| !label.starts_with("le="))
                .collect::<Vec<_>>()
                .join(",")
        }
        let mut out = String::new();
        let mut series: Vec<(String, Vec<&str>)> = Vec::new();
        for line in exposition.lines() {
            if line.starts_with('#') {
                flush(&mut out, &mut series);
                out.push_str(line);
                out.push('\n');
                continue;
            }
            let key = labels(line);
            match series.last_mut() {
                Some((last, lines)) if *last == key => lines.push(line),
                _ => series.push((key, vec![line])),
            }
        }
        flush(&mut out, &mut series);
        out
    }

    /// `metrics.txt` is the exposition of a gateway under the lifecycle policy before any traffic, every family the
    /// gateway exports, its series in the order of their labels: every metric's name, type, help and labels, which
    /// the dashboards' and alerts' checks (`ops/`) read. `FLATSHARE_GATEWAY_WRITE_METRICS=1` writes it anew (`make
    /// metrics`).
    #[test]
    fn the_checked_in_exposition_is_current() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("metrics.txt");
        let metrics = Metrics::with_harness(&harness::Metrics::every());
        // Requests by status and lifecycle events by type have no series before traffic, and a family without
        // one is left out of the exposition: one of each at zero, so that the file names them and their labels.
        create(
            &metrics.requests,
            RequestLabels {
                tenant: None,
                route: Route::Other.as_str(),
                class: None,
                status: 404,
            },
        );
        create(
            &metrics.events,
            EventLabels {
                r#type: "turn_started",
                source: "client",
                duplicate: false,
            },
        );
        let exposition = by_labels(&metrics.encode());
        if std::env::var_os("FLATSHARE_GATEWAY_WRITE_METRICS").is_some() {
            std::fs::write(&path, &exposition).unwrap();
        }
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            on_disk == exposition,
            "{} is stale; write it anew with `make metrics`",
            path.display()
        );
    }

    #[test]
    fn series_sort_by_their_labels_and_keep_their_own_lines_in_order() {
        let exposition = "# HELP h A histogram.\n# TYPE h histogram\n\
            h_sum{class=\"b\"} 0\nh_count{class=\"b\"} 0\nh_bucket{le=\"1.0\",class=\"b\"} 0\n\
            h_bucket{le=\"+Inf\",class=\"b\"} 0\n\
            h_sum{class=\"a\"} 0\nh_count{class=\"a\"} 0\nh_bucket{le=\"1.0\",class=\"a\"} 0\n\
            h_bucket{le=\"+Inf\",class=\"a\"} 0\n\
            # HELP c A counter.\n# TYPE c counter\nc_total 0\n# EOF\n";
        let sorted = by_labels(exposition);
        let lines: Vec<&str> = sorted.lines().collect();
        assert_eq!(
            lines[2..6].iter().map(|l| &l[..5]).collect::<Vec<_>>(),
            ["h_sum", "h_cou", "h_buc", "h_buc"]
        );
        assert!(lines[2].ends_with("{class=\"a\"} 0") && lines[6].ends_with("{class=\"b\"} 0"));
        assert_eq!(lines[4], "h_bucket{le=\"1.0\",class=\"a\"} 0");
        assert_eq!(
            &lines[10..],
            [
                "# HELP c A counter.",
                "# TYPE c counter",
                "c_total 0",
                "# EOF"
            ]
        );
        assert_eq!(by_labels(&sorted), sorted);
    }
}
