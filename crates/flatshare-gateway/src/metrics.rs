//! The gateway's Prometheus metrics, in a registry it owns (no global state), rendered at `GET /metrics`, and, with an
//! admission policy, the harness's after them (`harness`).
//!
//! Every label takes its values from a fixed set or from the configuration (tenant names), so the number of series
//! is bounded. Every series except those of requests (labeled with a status) and lifecycle events is created at
//! start, so it exports zero before traffic. The series a proxied request updates whose labels are fixed (its
//! class, its outcome, its route) are resolved then too, so that its bookkeeping takes no family's lock and hashes no
//! labels: of the families it touches, only requests by tenant and status are looked up per request.

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
    /// The routes that proxy a request body, whose overhead and upload are timed.
    const INFERENCE: [Self; 3] = [Self::ChatCompletions, Self::Completions, Self::Responses];

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

/// The kinds of `flatshare_gateway_usage_tokens_total`, in the order of [`Resolved::usage_tokens`].
const USAGE_KINDS: [&str; 3] = ["prompt", "completion", "cached"];

/// The series of fixed labels that every proxied request updates, each a handle to its family's series: by class
/// ([`SloClass::ALL`] order), outcome ([`Outcome::ALL`]), usage kind ([`USAGE_KINDS`]) or inference route
/// ([`Route::INFERENCE`]).
#[derive(Clone, Debug)]
struct Resolved {
    outcomes: [[Counter; Outcome::ALL.len()]; 3],
    first_byte: [Histogram; 3],
    duration: [Histogram; 3],
    streams_active: [Gauge; 3],
    in_flight: [Gauge; 3],
    usage_tokens: [[Counter; USAGE_KINDS.len()]; 3],
    overhead: [Histogram; Route::INFERENCE.len()],
    request_body: [Histogram; Route::INFERENCE.len()],
}

/// The index of `class` in [`Resolved`]'s arrays.
fn class_index(class: SloClass) -> usize {
    class as usize
}

/// The index of `route`, a route that proxies a body, in [`Resolved`]'s arrays.
fn inference_index(route: Route) -> usize {
    Route::INFERENCE
        .iter()
        .position(|&inference| inference == route)
        .expect("only the inference routes time a request body")
}

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
    in_flight: Family<ClassLabels, Gauge>,
    buffered_body_bytes: Gauge,
    predecessor_in_flight: Gauge,
    sanitized: Family<FieldLabels, Counter>,
    class_clamped: Counter,
    upstream_errors: Family<KindLabels, Counter>,
    error_events: Family<ErrorEventLabels, Counter>,
    events: Family<EventLabels, Counter>,
    usage_tokens: Family<UsageLabels, Counter>,
    /// Whether the engine is ready.
    node_ready: Gauge,
    resolved: Resolved,
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
        let outcomes: Family<OutcomeLabels, Counter> = Family::default();
        let overhead: Histograms<RouteLabels> =
            Family::new_with_constructor(|| Histogram::new(OVERHEAD_BUCKETS));
        let request_body: Histograms<RouteLabels> =
            Family::new_with_constructor(|| Histogram::new(OVERHEAD_BUCKETS));
        let first_byte: Histograms<ClassLabels> =
            Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS));
        let duration: Histograms<ClassLabels> =
            Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS));
        let streams_active: Family<ClassLabels, Gauge> = Family::default();
        let in_flight: Family<ClassLabels, Gauge> = Family::default();
        let usage_tokens: Family<UsageLabels, Counter> = Family::default();
        let by_class = SloClass::ALL.map(SloClass::as_str);
        let by_route = Route::INFERENCE.map(Route::as_str);
        // Each handle creates its series, at zero.
        let resolved = Resolved {
            outcomes: by_class.map(|class| {
                Outcome::ALL.map(|outcome| {
                    let outcome = outcome.as_str();
                    outcomes.get_or_create_owned(&OutcomeLabels { class, outcome })
                })
            }),
            first_byte: by_class
                .map(|class| first_byte.get_or_create_owned(&ClassLabels { class })),
            duration: by_class.map(|class| duration.get_or_create_owned(&ClassLabels { class })),
            streams_active: by_class
                .map(|class| streams_active.get_or_create_owned(&ClassLabels { class })),
            in_flight: by_class.map(|class| in_flight.get_or_create_owned(&ClassLabels { class })),
            usage_tokens: by_class.map(|class| {
                USAGE_KINDS
                    .map(|kind| usage_tokens.get_or_create_owned(&UsageLabels { class, kind }))
            }),
            overhead: by_route.map(|route| overhead.get_or_create_owned(&RouteLabels { route })),
            request_body: by_route
                .map(|route| request_body.get_or_create_owned(&RouteLabels { route })),
        };
        let metrics = Self {
            registry: Arc::new(Registry::default()),
            requests: Family::default(),
            outcomes,
            overhead,
            request_body,
            first_byte,
            duration,
            streams_active,
            in_flight,
            buffered_body_bytes: Gauge::default(),
            predecessor_in_flight: Gauge::default(),
            sanitized: Family::default(),
            class_clamped: Counter::default(),
            upstream_errors: Family::default(),
            error_events: Family::default(),
            events: Family::default(),
            usage_tokens,
            node_ready: Gauge::default(),
            resolved,
        };
        for class in by_class {
            for r#type in ErrorEvent::ALL.map(ErrorEvent::as_str) {
                create(&metrics.error_events, ErrorEventLabels { class, r#type });
            }
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
            "in_flight",
            "Inference requests past the intake, by class: from their headers to their end, reading their body, \
             waiting at the admission point or relayed",
            self.in_flight.clone(),
        );
        r.register(
            "buffered_body_bytes",
            "Bytes of request bodies the gateway holds, which max_buffered_body_bytes bounds",
            self.buffered_body_bytes.clone(),
        );
        r.register(
            "predecessor_in_flight",
            "A reload's predecessor's requests in flight, which the gateway counts with its own until the \
             predecessor exits",
            self.predecessor_in_flight.clone(),
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
        self.resolved.outcomes[class_index(class)][outcome as usize].inc();
    }

    /// How long the body of a request of `route`, an inference route, took to arrive after its headers.
    pub fn request_body(&self, route: Route, elapsed: Duration) {
        self.resolved.request_body[inference_index(route)].observe(elapsed.as_secs_f64());
    }

    /// The overhead histogram of `route`, an inference route, observed when the upstream request is written.
    pub fn overhead(&self, route: Route) -> Histogram {
        self.resolved.overhead[inference_index(route)].clone()
    }

    pub fn first_byte(&self, class: SloClass, elapsed: Duration) {
        self.resolved.first_byte[class_index(class)].observe(elapsed.as_secs_f64());
    }

    pub fn duration(&self, class: SloClass, elapsed: Duration) {
        self.resolved.duration[class_index(class)].observe(elapsed.as_secs_f64());
    }

    /// Counts a stream as active until the returned guard drops.
    pub fn stream_started(&self, class: SloClass) -> GaugeGuard {
        GaugeGuard::new(self.resolved.streams_active[class_index(class)].clone())
    }

    /// The gauge of whether the engine is ready.
    pub fn node_ready(&self) -> Gauge {
        self.node_ready.clone()
    }

    /// The intake's gauges.
    pub fn intake_gauges(&self) -> crate::intake::Gauges {
        crate::intake::Gauges {
            in_flight: self.resolved.in_flight.clone(),
            body_bytes: self.buffered_body_bytes.clone(),
            predecessor_in_flight: self.predecessor_in_flight.clone(),
        }
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
        let counters = &self.resolved.usage_tokens[class_index(class)];
        for (counter, tokens) in counters
            .iter()
            .zip([usage.prompt, usage.completion, usage.cached])
        {
            counter.inc_by(tokens);
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
    fn each_resolved_series_is_its_familys_series_of_its_labels() {
        // An update through a resolved handle shows under its labels, and only there: the indices follow the
        // label sets' orders.
        let metrics = Metrics::new();
        metrics.outcome(SloClass::Background, Outcome::Shed);
        let usage = Usage {
            prompt: 3,
            completion: 2,
            cached: 1,
        };
        metrics.usage(SloClass::Realtime, &usage);
        metrics.first_byte(SloClass::Interactive, Duration::from_millis(1));
        metrics.duration(SloClass::Background, Duration::from_millis(1));
        let _active = metrics.stream_started(SloClass::Realtime);
        metrics.overhead(Route::Responses).observe(1e-4);
        metrics.request_body(Route::Completions, Duration::from_micros(10));
        let text = metrics.encode();
        let value = |series: &str| -> f64 {
            text.lines()
                .find_map(|line| line.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
                .unwrap_or_else(|| panic!("no {series}"))
        };
        let changed = [
            (
                "flatshare_gateway_request_outcomes_total{class=\"background\",outcome=\"shed\"}",
                1.0,
            ),
            (
                "flatshare_gateway_usage_tokens_total{class=\"realtime\",kind=\"prompt\"}",
                3.0,
            ),
            (
                "flatshare_gateway_usage_tokens_total{class=\"realtime\",kind=\"completion\"}",
                2.0,
            ),
            (
                "flatshare_gateway_usage_tokens_total{class=\"realtime\",kind=\"cached\"}",
                1.0,
            ),
            (
                "flatshare_gateway_first_byte_seconds_count{class=\"interactive\"}",
                1.0,
            ),
            (
                "flatshare_gateway_duration_seconds_count{class=\"background\"}",
                1.0,
            ),
            ("flatshare_gateway_streams_active{class=\"realtime\"}", 1.0),
            (
                "flatshare_gateway_overhead_seconds_count{route=\"/v1/responses\"}",
                1.0,
            ),
            (
                "flatshare_gateway_request_body_seconds_count{route=\"/v1/completions\"}",
                1.0,
            ),
        ];
        for (series, expected) in changed {
            assert_eq!(value(series), expected, "{series}");
        }
        // Nothing else changed: the nine series above, and the sums of the four histograms among them.
        let nonzero = text
            .lines()
            .filter(|line| !line.starts_with('#') && !line.contains("_bucket{"))
            .filter(|line| {
                line.rsplit(' ')
                    .next()
                    .is_some_and(|v| v != "0" && v != "0.0")
            })
            .filter(|line| !line.starts_with("flatshare_gateway_node_ready"))
            .count();
        assert_eq!(nonzero, changed.len() + 4, "{text}");
    }

    /// What a proxied request's metric bookkeeping costs (its upload, overhead, stream, request, outcome, duration,
    /// first byte and usage), with one thread at it and with four at once. `cargo test --release -p
    /// flatshare-gateway --lib metrics -- --ignored --nocapture` prints it.
    #[test]
    #[ignore = "a benchmark, for --release"]
    fn what_a_requests_bookkeeping_costs() {
        const N: u32 = 200_000;
        let metrics = Metrics::new();
        let tenant: Arc<str> = "agents".into();
        let usage = Usage {
            prompt: 512,
            completion: 21,
            cached: 448,
        };
        let class = SloClass::Interactive;
        let request = || {
            metrics.request_body(Route::ChatCompletions, Duration::from_micros(3));
            metrics.overhead(Route::ChatCompletions).observe(30e-6);
            drop(metrics.stream_started(class));
            metrics.request(Some(&tenant), Route::ChatCompletions, Some(class), 200);
            metrics.outcome(class, Outcome::Completed);
            metrics.duration(class, Duration::from_micros(80));
            metrics.first_byte(class, Duration::from_micros(70));
            metrics.usage(class, &usage);
        };
        for threads in [1, 4] {
            let start = std::time::Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| (0..N).for_each(|_| request()));
                }
            });
            let ns = start.elapsed().as_nanos() as f64 / f64::from(N);
            println!(
                "a request's metric bookkeeping, {threads} thread(s) at once: {ns:.0} ns each"
            );
        }
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
