//! The gateway configuration: one TOML file, validated in full before anything binds. The crate README lists
//! every key; `deploy/cyan/gateway.example.toml` is a complete file.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use flatshare_proto::SloClass;
use http::uri::{Authority, Scheme, Uri};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// `[admission]`, which E2's harness reads (`harness`).
pub use crate::harness::config::Admission;
use crate::harness::config::RawAdmission;

/// A validated configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// The engine, spoken to over plain HTTP/1.1. With `[[nodes]]`, which name their own engines, the first node's.
    pub upstream: UpstreamAddr,
    /// The GPU nodes (`[[nodes]]`), each with its engine and speech servers, in the file's order; empty for a gateway
    /// with one `upstream`, which serves it as it always has.
    pub nodes: Vec<NodeConfig>,
    /// How sessions are placed on the nodes and the nodes judged (`[placement]`); read only with `[[nodes]]`.
    pub placement: Placement,
    /// How long an idle upstream connection is kept for reuse. It must be shorter than the engine's own
    /// keep-alive, or a request can be sent on a connection the engine is closing.
    pub upstream_idle_timeout: Duration,
    /// How a request's class reaches the upstream as a priority.
    pub upstream_priority: PrioritySignals,
    /// The admission point; `None` (policy "none") sends every request upstream at once.
    pub admission: Option<Admission>,
    pub listeners: Vec<Listen>,
    pub tenants: Vec<Tenant>,
    /// The key of every tenant's `cache_salt`, from `cache_salt_secret_file`.
    pub cache_salt_secret: CacheSaltSecret,
    /// What `GET /metrics` asks of a request on a TCP listener, from `metrics_key_file`; `None` without one,
    /// which only a gateway without TCP listeners may be.
    pub metrics_key: Option<MetricsKey>,
    /// Largest accepted request body; larger requests get 413.
    pub max_body_bytes: usize,
    /// What each class's requests may ask for (`[classes.<class>]`), by [`SloClass`] (strictest first; index a class
    /// with [`Config::class`]).
    pub classes: [ClassConfig; 3],
    /// The inference canary; `None` when `[canary] enabled = false`, as E2's runs set it.
    pub canary: Option<Canary>,
    /// The choices a request may ask for at most: its `n` (with `use_beam_search`, its beam's width) and its
    /// `best_of`. A request that asks for more gets 400.
    pub max_choices: u32,
    /// The inference requests in flight at most, from their headers to their end: one more gets 503 at once, before
    /// its body is read (`crate::intake`).
    pub max_in_flight: usize,
    /// The requests waiting at the admission point at most, in all its queues; one more gets 429.
    pub max_queued: usize,
    /// The bytes of request bodies the gateway holds at most, in all: a request whose body would pass it gets 503.
    pub max_buffered_body_bytes: usize,
    /// Frames buffered between the upstream and a client before backpressure reaches the upstream.
    pub stream_buffer_frames: usize,
    /// How long a full stream buffer may stay full before the stream is aborted as stalled.
    pub stall_timeout: Duration,
    /// Every class's default `headers_timeout_s`: how long an attempt waits for the upstream's response headers.
    pub first_byte_timeout: Duration,
    /// How long a shutdown waits for in-flight requests.
    pub drain_timeout: Duration,
    /// How far into a drain background requests go on before the gateway cuts them, so that a stop or a reload
    /// is not held up by the longest agent turn: a stream then ends with a `draining` error event, and a request
    /// without a response yet gets 503. Realtime and interactive requests have the whole drain timeout.
    pub background_drain: Duration,
    /// Lifecycle-event ids remembered for deduplication, and for how long.
    pub event_dedup_capacity: usize,
    pub event_dedup_ttl: Duration,
    pub log_format: LogFormat,
    pub log_level: LogLevel,
}

impl Config {
    /// The configuration of `class`'s requests.
    pub fn class(&self, class: SloClass) -> &ClassConfig {
        &self.classes[class as usize]
    }
}

/// `[classes.<class>]`: what the requests of one class may ask for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassConfig {
    /// The cap on the class's replies, in tokens (`max_reply_tokens`; for realtime, `realtime_max_tokens` too): a
    /// request's body gets its route's own limit at the lower of the client's limit and the cap
    /// ([`crate::bounds`]), as degrade caps a reply under the lifecycle policy. `None` caps no reply.
    pub max_reply_tokens: Option<u32>,
    /// The class's requests in flight at most: one more gets 503 at once, before its body is read.
    pub max_in_flight: usize,
    /// The class's deadlines ([`crate::deadline`]): its body's upload, a wait at the admission point, an attempt's
    /// response headers, the first content and the whole request from the receipt, and the wait between two frames.
    pub upload_timeout: Duration,
    pub queue_timeout: Duration,
    pub headers_timeout: Duration,
    pub first_content_timeout: Duration,
    pub idle_timeout: Duration,
    pub total_timeout: Duration,
    /// The service level's objective for the class's first content, from the receipt
    /// (`first_content_objective_s`): a valid request is good when it completes with its first content within it.
    /// `None`: completing is enough.
    pub first_content_objective: Option<Duration>,
}

impl ClassConfig {
    /// A class's defaults. The classes' in-flight bounds leave the other classes room under `max_in_flight`'s
    /// default of 1,024: interactive and background requests together reach 768 at most, so a flood of them cannot
    /// take the realtime class's 128. E2's heaviest point held about 30 in flight (24 agent sessions and voice).
    ///
    /// The deadlines are a few times what the recorded runs took at most, so that none ends a request they made:
    /// background first bytes ran to 232 s at p99 (W5's resubmitting arm), behind preemptions, realtime ones to 13.4 s
    /// (the worst arm's p99), and no run met the 600 s first-byte timeout. A voice reply ran 52 tokens at most.
    pub(crate) fn defaults(class: SloClass) -> Self {
        let realtime = class == SloClass::Realtime;
        let s = Duration::from_secs;
        Self {
            max_reply_tokens: None,
            max_in_flight: match class {
                SloClass::Realtime => 128,
                SloClass::Interactive => 256,
                SloClass::Background => 512,
            },
            upload_timeout: s(60),
            queue_timeout: s(120),
            headers_timeout: s(600),
            first_content_timeout: s(if realtime { 60 } else { 600 }),
            idle_timeout: s(if realtime { 30 } else { 600 }),
            total_timeout: s(match class {
                SloClass::Realtime => 300,
                SloClass::Interactive => 1800,
                SloClass::Background => 3600,
            }),
            // docs/SLO.md's realtime first byte: 95% within 250 ms.
            first_content_objective: realtime.then(|| Duration::from_millis(250)),
        }
    }
}

/// `[canary]`: the inference canary ([`crate::canary`]), a one-token generation at the lowest class's priority, sent
/// only when no inference succeeded within `interval`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Canary {
    pub interval: Duration,
    pub timeout: Duration,
    /// The consecutive failures, with nothing succeeding between them, that make the engine degraded.
    pub degraded_after: u32,
}

/// One `[[nodes]]` entry: a GPU node, its engine and its speech servers, as cyan's tunnels reach them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeConfig {
    /// Its name, in `/nodes`, the metrics' `node` label and `POST /v1/sessions`' answers.
    pub name: String,
    /// Its engine.
    pub upstream: UpstreamAddr,
    /// Whether it takes realtime sessions (calls), and interactive and background ones (agents).
    pub realtime: bool,
    pub background: bool,
    /// The realtime sessions placed on it at most: a call each.
    pub max_calls: usize,
    /// The agents' sessions (interactive and background) placed on it at most, `None` for no bound: what a node
    /// that carries calls may run beside them.
    pub max_agents: Option<usize>,
    /// Its speech servers, whose `/health` its health needs, and which a call placed on it binds to.
    pub stt: Option<UpstreamAddr>,
    pub tts: Option<UpstreamAddr>,
}

/// `[placement]`: how sessions are placed on the nodes, and when a node is healthy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// How long a session keeps its node without a request (`idle_s`).
    pub idle: Duration,
    /// How often each node's engine and speech servers are checked (`health_interval_s`), and how long a speech
    /// server's `/health` may take (`health_timeout_s`).
    pub health_interval: Duration,
    pub health_timeout: Duration,
    /// The failed checks in a row that make a healthy node unhealthy (`unhealthy_after`), and the passed ones that
    /// make an unhealthy node healthy again (`healthy_after`).
    pub unhealthy_after: u32,
    pub healthy_after: u32,
    /// The interactive and background requests in flight at most on a node that carries realtime sessions moved
    /// from a node that is not healthy (`failover_background_max_in_flight`): the others wait.
    pub failover_background_max_in_flight: usize,
}

impl Default for Placement {
    fn default() -> Self {
        Self {
            // The 5 minutes a prefix cache is worth keeping a session for; E2's agents think for 2 minutes at most,
            // and a call's sessions end with its session_closed event.
            idle: Duration::from_secs(300),
            health_interval: Duration::from_secs(1),
            health_timeout: Duration::from_secs(1),
            unhealthy_after: 3,
            healthy_after: 2,
            failover_background_max_in_flight: 0,
        }
    }
}

/// One listener.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listen {
    pub addr: ListenAddr,
    /// The tenant that requests without a known API key belong to. Only Unix sockets may name one: their file
    /// mode keeps other users out, while any local user can reach a loopback TCP port.
    pub tenant: Option<String>,
}

/// Where a listener binds: a loopback TCP address or a Unix socket path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ListenAddr {
    Tcp(SocketAddr),
    Unix(PathBuf),
}

/// Where an engine listens: `http://host:port`, or `unix:` and the path of its socket (the engine's own, or a
/// tunnel's on this host).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpstreamAddr {
    Tcp(Authority),
    Unix(PathBuf),
}

impl fmt::Display for UpstreamAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tcp(authority) => write!(f, "http://{authority}"),
            Self::Unix(path) => write!(f, "unix:{}", path.display()),
        }
    }
}

/// A tenant: who a request belongs to, what it may ask for, and its cache namespace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tenant {
    pub name: String,
    /// SHA-256 digests of the tenant's API keys. The keys themselves never appear in the configuration.
    pub keys_sha256: Vec<[u8; 32]>,
    /// A trusted tenant may set `X-Session-ID`, choose its class with `X-Flatshare-Class` and post lifecycle events.
    pub trusted: bool,
    pub default_class: SloClass,
    /// The strictest class the tenant may use; a stricter request is clamped to it.
    pub max_class: SloClass,
    /// The tenant's requests in flight at most: one more gets 429 at once, before its body is read.
    pub max_in_flight: usize,
}

/// The secret the tenants' cache salts are derived from: 32 bytes, which its `Debug` does not show.
#[derive(Clone, PartialEq, Eq)]
pub struct CacheSaltSecret(pub(crate) [u8; 32]);

impl fmt::Debug for CacheSaltSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CacheSaltSecret(..)")
    }
}

/// The SHA-256 of the metrics key, which its `Debug` does not show.
#[derive(Clone, PartialEq, Eq)]
pub struct MetricsKey(pub(crate) [u8; 32]);

impl fmt::Debug for MetricsKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MetricsKey(..)")
    }
}

/// `upstream_priority`: where the class's priority goes in each upstream request. Engines and routers read it
/// in different places, and a stock deployment sends none, so each is a choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrioritySignals {
    /// `X-Vllm-Priority`, the class's engine priority (realtime 0, interactive 1, background 2), which vLLM's
    /// priority scheduling reads.
    pub vllm_header: bool,
    /// The same number as the body's `priority`, which SGLang's priority scheduling reads.
    pub body: bool,
    /// `X-SMG-Priority`: `system`, `interactive` or `bulk`, which SMG's priority scheduler reads.
    pub smg_header: bool,
}

/// One entry of `upstream_priority`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum PrioritySignal {
    VllmHeader,
    Body,
    SmgHeader,
}

/// `log_format`: human-readable lines, or one JSON object per line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Text,
    Json,
}

/// `log_level`: the most verbose level logged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Why a configuration was rejected.
#[derive(Debug)]
pub enum ConfigError {
    /// The file could not be read.
    Read(PathBuf, std::io::Error),
    /// The file is not TOML of the configuration's shape (an unknown key, a wrong type).
    Parse(PathBuf, toml::de::Error),
    /// A value the configuration's rules reject, such as a tenant no request can reach.
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(path, err) => write!(f, "cannot read {}: {err}", path.display()),
            Self::Parse(path, err) => write!(f, "{}: {err}", path.display()),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(_, e) => Some(e),
            Self::Parse(_, e) => Some(e),
            Self::Invalid(_) => None,
        }
    }
}

/// The file as written; a missing key takes its value from [`RawConfig::default`].
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawConfig {
    upstream: String,
    nodes: Vec<RawNode>,
    placement: Option<RawPlacement>,
    upstream_idle_timeout_s: f64,
    upstream_priority: Vec<PrioritySignal>,
    admission: RawAdmission,
    listen: Vec<RawListen>,
    tenants: BTreeMap<String, RawTenant>,
    cache_salt_secret_file: Option<PathBuf>,
    metrics_key_file: Option<PathBuf>,
    max_body_bytes: usize,
    realtime_max_tokens: Option<u32>,
    classes: RawClasses,
    canary: RawCanary,
    max_choices: u32,
    max_in_flight: usize,
    max_queued: usize,
    max_buffered_body_bytes: usize,
    stream_buffer_frames: usize,
    stall_timeout_s: f64,
    first_byte_timeout_s: f64,
    drain_timeout_s: f64,
    background_drain_s: f64,
    event_dedup_capacity: usize,
    event_dedup_ttl_s: f64,
    log_format: LogFormat,
    log_level: LogLevel,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            upstream: String::new(),
            nodes: Vec::new(),
            placement: None,
            // vLLM closes idle connections after VLLM_HTTP_TIMEOUT_KEEP_ALIVE, 5 s by default.
            upstream_idle_timeout_s: 4.0,
            upstream_priority: vec![PrioritySignal::VllmHeader],
            admission: RawAdmission::default(),
            listen: Vec::new(),
            tenants: BTreeMap::new(),
            cache_salt_secret_file: None,
            metrics_key_file: None,
            max_body_bytes: 32 << 20,
            realtime_max_tokens: None,
            classes: RawClasses::default(),
            canary: RawCanary::default(),
            max_choices: 1,
            // Envoy's circuit breakers' default for requests in flight.
            max_in_flight: 1024,
            max_queued: 256,
            // Eight bodies of max_body_bytes' 32 MiB; E2's agent requests run to about 130 KB each.
            max_buffered_body_bytes: 256 << 20,
            stream_buffer_frames: 64,
            stall_timeout_s: 30.0,
            first_byte_timeout_s: 600.0,
            drain_timeout_s: 30.0,
            background_drain_s: 5.0,
            event_dedup_capacity: 100_000,
            event_dedup_ttl_s: 600.0,
            log_format: LogFormat::Text,
            log_level: LogLevel::Info,
        }
    }
}

/// `[classes]`: a table for each class that sets anything.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawClasses {
    realtime: RawClass,
    interactive: RawClass,
    background: RawClass,
}

/// `[classes.<class>]` as written; a missing key takes the class's default.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawClass {
    max_reply_tokens: Option<u32>,
    max_in_flight: Option<usize>,
    upload_timeout_s: Option<f64>,
    queue_timeout_s: Option<f64>,
    headers_timeout_s: Option<f64>,
    first_content_timeout_s: Option<f64>,
    idle_timeout_s: Option<f64>,
    total_timeout_s: Option<f64>,
    first_content_objective_s: Option<f64>,
}

/// `[canary]` as written.
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawCanary {
    enabled: bool,
    interval_s: f64,
    timeout_s: f64,
    degraded_after: u32,
}

impl Default for RawCanary {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_s: 30.0,
            // A canary waits at the lowest priority behind every class's prefill.
            timeout_s: 60.0,
            degraded_after: 3,
        }
    }
}

impl RawCanary {
    fn validate(self) -> Result<Option<Canary>, String> {
        // Its other keys may stay set while it is off: E2's runs switch it off over the deployment's table.
        if !self.enabled {
            return Ok(None);
        }
        if self.degraded_after == 0 {
            return Err("canary.degraded_after must be positive".into());
        }
        Ok(Some(Canary {
            interval: seconds("canary.interval_s", self.interval_s)?,
            timeout: seconds("canary.timeout_s", self.timeout_s)?,
            degraded_after: self.degraded_after,
        }))
    }
}

/// A role a node takes: the classes whose sessions it is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    /// Realtime sessions: calls.
    Realtime,
    /// Interactive and background sessions: agents.
    Background,
}

/// A `[[nodes]]` entry as written.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNode {
    name: String,
    upstream: String,
    #[serde(default = "RawNode::every_role")]
    roles: Vec<Role>,
    /// E2 measured two calls on one card.
    #[serde(default = "RawNode::two_calls")]
    max_calls: usize,
    /// No bound unless set.
    #[serde(default)]
    max_agents: Option<usize>,
    #[serde(default)]
    stt: Option<String>,
    #[serde(default)]
    tts: Option<String>,
}

impl RawNode {
    fn every_role() -> Vec<Role> {
        vec![Role::Realtime, Role::Background]
    }

    fn two_calls() -> usize {
        2
    }
}

/// `[placement]` as written; a missing key takes [`Placement::default`]'s value.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawPlacement {
    idle_s: Option<f64>,
    health_interval_s: Option<f64>,
    health_timeout_s: Option<f64>,
    unhealthy_after: Option<u32>,
    healthy_after: Option<u32>,
    failover_background_max_in_flight: Option<usize>,
}

impl RawPlacement {
    fn validate(self) -> Result<Placement, String> {
        let defaults = Placement::default();
        let duration = |key: &str, value: Option<f64>, default: Duration| match value {
            Some(value) => seconds(&format!("placement.{key}"), value),
            None => Ok(default),
        };
        let count = |key: &str, value: Option<u32>, default: u32| match value {
            Some(0) => Err(format!("placement.{key} must be positive")),
            Some(value) => Ok(value),
            None => Ok(default),
        };
        Ok(Placement {
            idle: duration("idle_s", self.idle_s, defaults.idle)?,
            health_interval: duration(
                "health_interval_s",
                self.health_interval_s,
                defaults.health_interval,
            )?,
            health_timeout: duration(
                "health_timeout_s",
                self.health_timeout_s,
                defaults.health_timeout,
            )?,
            unhealthy_after: count(
                "unhealthy_after",
                self.unhealthy_after,
                defaults.unhealthy_after,
            )?,
            healthy_after: count("healthy_after", self.healthy_after, defaults.healthy_after)?,
            failover_background_max_in_flight: self
                .failover_background_max_in_flight
                .unwrap_or(defaults.failover_background_max_in_flight),
        })
    }
}

/// The `[[nodes]]` entries: names that metric labels carry, each once, no engine twice, and every role taken by
/// some node, since a class whose role no node takes could never be placed.
fn validate_nodes(raw: Vec<RawNode>) -> Result<Vec<NodeConfig>, String> {
    let mut nodes: Vec<NodeConfig> = Vec::with_capacity(raw.len());
    for (i, node) in raw.into_iter().enumerate() {
        if !valid_name(&node.name) {
            return Err(format!(
                "nodes[{i}].name {:?}: use 1 to 64 characters from [A-Za-z0-9._-]",
                node.name
            ));
        }
        let name = node.name;
        if nodes.iter().any(|other| other.name == name) {
            return Err(format!("nodes: {name:?} is named twice"));
        }
        let upstream = parse_address(&format!("nodes.{name}.upstream"), &node.upstream)?;
        if let Some(other) = nodes.iter().find(|other| other.upstream == upstream) {
            return Err(format!(
                "nodes.{name}.upstream is nodes.{}'s: one engine would count as two nodes",
                other.name
            ));
        }
        let mut roles = [false; 2];
        for role in &node.roles {
            if std::mem::replace(&mut roles[*role as usize], true) {
                return Err(format!("nodes.{name}.roles names a role twice"));
            }
        }
        let [realtime, background] = roles;
        if !realtime && !background {
            return Err(format!(
                "nodes.{name}.roles is empty: name realtime, background or both"
            ));
        }
        if realtime && node.max_calls == 0 {
            return Err(format!(
                "nodes.{name}.max_calls must be positive for a node with the realtime role"
            ));
        }
        match node.max_agents {
            Some(_) if !background => {
                return Err(format!(
                    "nodes.{name}.max_agents is for a node with the background role: this one takes no agents' \
                     sessions"
                ));
            }
            Some(0) => {
                return Err(format!(
                    "nodes.{name}.max_agents must be positive, or left out for no bound"
                ));
            }
            _ => {}
        }
        let speech = |key: &str, value: Option<String>| {
            value
                .map(|raw| parse_address(&format!("nodes.{name}.{key}"), &raw))
                .transpose()
        };
        let stt = speech("stt", node.stt)?;
        let tts = speech("tts", node.tts)?;
        nodes.push(NodeConfig {
            upstream,
            realtime,
            background,
            max_calls: node.max_calls,
            max_agents: node.max_agents,
            stt,
            tts,
            name,
        });
    }
    for (role, takes) in [
        ("realtime", nodes.iter().any(|node| node.realtime)),
        ("background", nodes.iter().any(|node| node.background)),
    ] {
        if !nodes.is_empty() && !takes {
            return Err(format!(
                "no node has the {role} role, so no {role} session could be placed"
            ));
        }
    }
    Ok(nodes)
}

impl RawClasses {
    /// Each class's configuration, strictest first. `realtime_max_tokens` is the realtime class's
    /// `max_reply_tokens`, from before there were classes' tables, and only one of the two may be set;
    /// `first_byte_timeout` is every class's default `headers_timeout_s`.
    fn validate(
        self,
        realtime_max_tokens: Option<u32>,
        first_byte_timeout: Duration,
    ) -> Result<[ClassConfig; 3], String> {
        let Self {
            mut realtime,
            interactive,
            background,
        } = self;
        match (realtime_max_tokens, realtime.max_reply_tokens) {
            (Some(0), _) => return Err("realtime_max_tokens must be positive".into()),
            (Some(_), Some(_)) => {
                return Err(
                    "realtime_max_tokens and classes.realtime.max_reply_tokens are the same cap: set one".into(),
                );
            }
            (Some(cap), None) => realtime.max_reply_tokens = Some(cap),
            (None, _) => {}
        }
        let mut classes = SloClass::ALL.map(ClassConfig::defaults);
        for (class, raw) in SloClass::ALL
            .into_iter()
            .zip([realtime, interactive, background])
        {
            let config = &mut classes[class as usize];
            for (key, zero) in [
                ("max_reply_tokens", raw.max_reply_tokens == Some(0)),
                ("max_in_flight", raw.max_in_flight == Some(0)),
            ] {
                if zero {
                    return Err(format!("classes.{class}.{key} must be positive"));
                }
            }
            config.max_reply_tokens = raw.max_reply_tokens;
            config.max_in_flight = raw.max_in_flight.unwrap_or(config.max_in_flight);
            config.headers_timeout = first_byte_timeout;
            for (key, set, timeout) in [
                (
                    "upload_timeout_s",
                    raw.upload_timeout_s,
                    &mut config.upload_timeout,
                ),
                (
                    "queue_timeout_s",
                    raw.queue_timeout_s,
                    &mut config.queue_timeout,
                ),
                (
                    "headers_timeout_s",
                    raw.headers_timeout_s,
                    &mut config.headers_timeout,
                ),
                (
                    "first_content_timeout_s",
                    raw.first_content_timeout_s,
                    &mut config.first_content_timeout,
                ),
                (
                    "idle_timeout_s",
                    raw.idle_timeout_s,
                    &mut config.idle_timeout,
                ),
                (
                    "total_timeout_s",
                    raw.total_timeout_s,
                    &mut config.total_timeout,
                ),
            ] {
                if let Some(value) = set {
                    *timeout = seconds(&format!("classes.{class}.{key}"), value)?;
                }
            }
            if let Some(objective) = raw.first_content_objective_s {
                let key = format!("classes.{class}.first_content_objective_s");
                config.first_content_objective = Some(seconds(&key, objective)?);
            }
        }
        Ok(classes)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawListen {
    tcp: Option<SocketAddr>,
    unix: Option<PathBuf>,
    tenant: Option<String>,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawTenant {
    keys_sha256: Vec<String>,
    trusted: bool,
    default_class: SloClass,
    /// Defaults to `default_class`: a tenant gets no stricter class unless the file says so.
    max_class: Option<SloClass>,
    max_in_flight: usize,
}

impl Default for RawTenant {
    fn default() -> Self {
        Self {
            keys_sha256: Vec::new(),
            trusted: false,
            default_class: SloClass::default(),
            max_class: None,
            // Half of max_in_flight's default: one tenant's flood leaves the others room.
            max_in_flight: 512,
        }
    }
}

impl Config {
    /// Reads and validates a configuration file.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text =
            std::fs::read_to_string(path).map_err(|e| ConfigError::Read(path.to_owned(), e))?;
        let raw: RawConfig =
            toml::from_str(&text).map_err(|e| ConfigError::Parse(path.to_owned(), e))?;
        raw.validate()
            .map_err(|message| ConfigError::Invalid(format!("{}: {message}", path.display())))
    }

    /// Parses and validates a configuration from TOML text.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig =
            toml::from_str(text).map_err(|e| ConfigError::Parse(PathBuf::from("<config>"), e))?;
        raw.validate().map_err(ConfigError::Invalid)
    }
}

impl RawConfig {
    fn validate(self) -> Result<Config, String> {
        let nodes = validate_nodes(self.nodes)?;
        let upstream = match (nodes.first(), self.upstream.is_empty()) {
            (None, _) => parse_upstream(&self.upstream)?,
            (Some(first), true) => first.upstream.clone(),
            (Some(_), false) => {
                return Err(
                    "set upstream or [[nodes]], not both: each node names its own engine".into(),
                );
            }
        };
        let placement = match (self.placement, nodes.is_empty()) {
            (Some(_), true) => return Err("[placement] needs [[nodes]]".into()),
            (Some(raw), false) => raw.validate()?,
            (None, _) => Placement::default(),
        };
        let upstream_priority = priority_signals(&self.upstream_priority)?;
        let admission = self.admission.validate()?;
        if admission.is_some() && !nodes.is_empty() {
            return Err(
                "an [admission] policy is for one upstream: with [[nodes]] each engine would need an admission \
                 point of its own"
                    .into(),
            );
        }

        let mut tenants = Vec::with_capacity(self.tenants.len());
        let mut digests = HashSet::new();
        for (name, raw) in self.tenants {
            let tenant = raw.validate(name)?;
            for digest in &tenant.keys_sha256 {
                if !digests.insert(*digest) {
                    return Err(format!(
                        "tenants.{}: an API key digest is listed twice",
                        tenant.name
                    ));
                }
            }
            tenants.push(tenant);
        }
        if tenants.is_empty() {
            return Err("no [tenants.<name>] table".into());
        }

        let mut listeners = Vec::with_capacity(self.listen.len());
        for (i, raw) in self.listen.into_iter().enumerate() {
            let addr = match (raw.tcp, raw.unix) {
                (Some(addr), None) if addr.ip().is_loopback() => ListenAddr::Tcp(addr),
                (Some(addr), None) => {
                    return Err(format!(
                        "listen[{i}]: tcp {addr} is not a loopback address: the gateway has no TLS, so API keys \
                         would cross the network in cleartext"
                    ));
                }
                (None, Some(path)) if path.is_absolute() => ListenAddr::Unix(path),
                (None, Some(path)) => {
                    return Err(format!(
                        "listen[{i}]: unix path {} is not absolute",
                        path.display()
                    ));
                }
                _ => return Err(format!("listen[{i}]: set exactly one of tcp or unix")),
            };
            if let Some(tenant) = &raw.tenant {
                if !tenants.iter().any(|t| &t.name == tenant) {
                    return Err(format!("listen[{i}]: unknown tenant {tenant:?}"));
                }
                if matches!(addr, ListenAddr::Tcp(_)) {
                    return Err(format!(
                        "listen[{i}]: a TCP listener cannot name a tenant, because other users of the host can \
                         reach it; give the tenant an API key instead"
                    ));
                }
            }
            listeners.push(Listen {
                addr,
                tenant: raw.tenant,
            });
        }
        if listeners.is_empty() {
            return Err("no [[listen]] entry".into());
        }
        for tenant in &tenants {
            let reachable = !tenant.keys_sha256.is_empty()
                || listeners
                    .iter()
                    .any(|l| l.tenant.as_ref() == Some(&tenant.name));
            if !reachable {
                return Err(format!(
                    "tenants.{}: no API key and no listener names it, so no request can reach it",
                    tenant.name
                ));
            }
        }
        let cache_salt_secret = cache_salt_secret(self.cache_salt_secret_file)?;
        let tcp = listeners
            .iter()
            .any(|l| matches!(l.addr, ListenAddr::Tcp(_)));
        let metrics_key = metrics_key(self.metrics_key_file, tcp)?;

        if self.max_body_bytes == 0 {
            return Err("max_body_bytes must be positive".into());
        }
        let first_byte_timeout = seconds("first_byte_timeout_s", self.first_byte_timeout_s)?;
        let classes = self
            .classes
            .validate(self.realtime_max_tokens, first_byte_timeout)?;
        for (key, zero) in [
            ("max_choices", self.max_choices == 0),
            ("max_in_flight", self.max_in_flight == 0),
            ("max_queued", self.max_queued == 0),
        ] {
            if zero {
                return Err(format!("{key} must be positive"));
            }
        }
        if self.max_buffered_body_bytes < self.max_body_bytes {
            return Err(format!(
                "max_buffered_body_bytes ({}) must be at least max_body_bytes ({}), or no body of that size could \
                 be read",
                self.max_buffered_body_bytes, self.max_body_bytes
            ));
        }
        if self.stream_buffer_frames == 0 {
            return Err("stream_buffer_frames must be positive".into());
        }
        if self.event_dedup_capacity == 0 {
            return Err("event_dedup_capacity must be positive".into());
        }
        let drain_timeout = seconds("drain_timeout_s", self.drain_timeout_s)?;
        let background_drain = Duration::try_from_secs_f64(self.background_drain_s)
            .ok()
            .filter(|grace| *grace <= drain_timeout)
            .ok_or_else(|| {
                format!(
                    "background_drain_s must be a number of seconds from 0 to drain_timeout_s ({})",
                    self.drain_timeout_s
                )
            })?;
        Ok(Config {
            upstream,
            nodes,
            placement,
            upstream_idle_timeout: seconds(
                "upstream_idle_timeout_s",
                self.upstream_idle_timeout_s,
            )?,
            upstream_priority,
            admission,
            listeners,
            tenants,
            cache_salt_secret,
            metrics_key,
            max_body_bytes: self.max_body_bytes,
            classes,
            canary: self.canary.validate()?,
            max_choices: self.max_choices,
            max_in_flight: self.max_in_flight,
            max_queued: self.max_queued,
            max_buffered_body_bytes: self.max_buffered_body_bytes,
            stream_buffer_frames: self.stream_buffer_frames,
            stall_timeout: seconds("stall_timeout_s", self.stall_timeout_s)?,
            first_byte_timeout,
            drain_timeout,
            background_drain,
            event_dedup_capacity: self.event_dedup_capacity,
            event_dedup_ttl: seconds("event_dedup_ttl_s", self.event_dedup_ttl_s)?,
            log_format: self.log_format,
            log_level: self.log_level,
        })
    }
}

impl RawTenant {
    fn validate(self, name: String) -> Result<Tenant, String> {
        if !valid_name(&name) {
            return Err(format!(
                "tenant name {name:?}: use 1 to 64 characters from [A-Za-z0-9._-]"
            ));
        }
        let max_class = self.max_class.unwrap_or(self.default_class);
        // Classes order from strictest (realtime) to most lenient (background).
        if self.default_class < max_class {
            return Err(format!(
                "tenants.{name}: default_class {} is stricter than max_class {max_class}",
                self.default_class
            ));
        }
        let keys_sha256 = self
            .keys_sha256
            .iter()
            .map(|hex| {
                parse_digest(hex).ok_or_else(|| {
                    format!("tenants.{name}: keys_sha256 entries must be 64 hex characters")
                })
            })
            .collect::<Result<_, _>>()?;
        if self.max_in_flight == 0 {
            return Err(format!("tenants.{name}.max_in_flight must be positive"));
        }
        Ok(Tenant {
            name,
            keys_sha256,
            trusted: self.trusted,
            default_class: self.default_class,
            max_class,
            max_in_flight: self.max_in_flight,
        })
    }
}

/// A tenant name: 1 to 64 characters from `[A-Za-z0-9._-]`, which metric labels can carry.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn parse_upstream(raw: &str) -> Result<UpstreamAddr, String> {
    parse_address("upstream", raw)
}

/// An engine's or a speech server's address, `key` naming it in errors.
fn parse_address(key: &str, raw: &str) -> Result<UpstreamAddr, String> {
    const EXPECTED: &str =
        "expected http://host:port with no path, or unix: and an absolute socket path";
    if let Some(path) = raw.strip_prefix("unix:") {
        let path = Path::new(path);
        // One spelling: `unix://host/...` reads as a URL with an authority, which a socket has not.
        if !path.is_absolute() || path.as_os_str().as_encoded_bytes().starts_with(b"//") {
            return Err(format!("{key} {raw:?}: {EXPECTED}"));
        }
        // A path longer than a socket address holds would fail every connection; refuse it at start.
        if std::os::unix::net::SocketAddr::from_pathname(path).is_err() {
            return Err(format!(
                "{key} {raw:?}: the path is too long for a Unix socket address"
            ));
        }
        return Ok(UpstreamAddr::Unix(path.to_owned()));
    }
    let uri: Uri = raw
        .parse()
        .map_err(|e| format!("{key} {raw:?}: {e}; {EXPECTED}"))?;
    let path_is_root = matches!(uri.path(), "" | "/") && uri.query().is_none();
    match (uri.scheme(), uri.authority()) {
        (Some(scheme), Some(authority)) if *scheme == Scheme::HTTP && path_is_root => {
            Ok(UpstreamAddr::Tcp(authority.clone()))
        }
        _ => Err(format!("{key} {raw:?}: {EXPECTED}")),
    }
}

fn priority_signals(listed: &[PrioritySignal]) -> Result<PrioritySignals, String> {
    let mut signals = PrioritySignals::default();
    for signal in listed {
        let (set, name) = match signal {
            PrioritySignal::VllmHeader => (&mut signals.vllm_header, "vllm-header"),
            PrioritySignal::Body => (&mut signals.body, "body"),
            PrioritySignal::SmgHeader => (&mut signals.smg_header, "smg-header"),
        };
        if std::mem::replace(set, true) {
            return Err(format!("upstream_priority lists {name} twice"));
        }
    }
    Ok(signals)
}

/// Reads `cache_salt_secret_file`: 32 random bytes in hex, in a file only its owner may open. A salt anyone could
/// compute from the tenant's name would let a client that reaches the engine directly probe another tenant's
/// prefix cache.
fn cache_salt_secret(path: Option<PathBuf>) -> Result<CacheSaltSecret, String> {
    let path = path.ok_or(
        "no cache_salt_secret_file: name a file of 32 random bytes in hex, mode 600, from which the tenants' \
         cache salts are derived",
    )?;
    let key = "cache_salt_secret_file";
    parse_digest(private_file(key, &path)?.trim())
        .map(CacheSaltSecret)
        .ok_or_else(|| {
            format!(
                "{key} {}: expected 64 hex characters, 32 random bytes",
                path.display()
            )
        })
}

/// Reads `metrics_key_file`: one key on one line, in a file only its owner may open, whose SHA-256 `GET /metrics`
/// compares with a request's bearer key on a TCP listener. Every user of the host reaches loopback, and the
/// metrics name the tenants and their traffic, so a gateway with a TCP listener needs one.
fn metrics_key(path: Option<PathBuf>, tcp: bool) -> Result<Option<MetricsKey>, String> {
    let Some(path) = path else {
        return if tcp {
            Err(
                "no metrics_key_file: a TCP listener serves /metrics to every user of the host; name a file of \
                 one key, mode 600, that GET /metrics asks for there"
                    .into(),
            )
        } else {
            Ok(None)
        };
    };
    let key = "metrics_key_file";
    let text = private_file(key, &path)?;
    let token = text.trim();
    if token.is_empty() || token.contains(char::is_whitespace) {
        return Err(format!(
            "{key} {}: expected one key on one line",
            path.display()
        ));
    }
    Ok(Some(MetricsKey(Sha256::digest(token).into())))
}

/// The text of a file that only its owner may open, named by the configuration's `key` by an absolute path.
fn private_file(key: &str, path: &Path) -> Result<String, String> {
    let fail = |cause: String| format!("{key} {}: {cause}", path.display());
    if !path.is_absolute() {
        return Err(fail("not an absolute path".into()));
    }
    let mode = std::fs::metadata(path)
        .map_err(|e| fail(e.to_string()))?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err(fail(format!(
            "mode {:o} lets other users open it; make it 600",
            mode & 0o777
        )));
    }
    std::fs::read_to_string(path).map_err(|e| fail(e.to_string()))
}

fn parse_digest(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut digest = [0u8; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(digest)
}

pub(crate) fn seconds(key: &str, value: f64) -> Result<Duration, String> {
    Duration::try_from_secs_f64(value)
        .ok()
        .filter(|d| !d.is_zero())
        .ok_or_else(|| format!("{key} must be a positive number of seconds"))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use super::*;

    /// SHA-256 of "foo".
    const KEY: &str = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae";

    /// The bytes 0 to 31, in hex.
    const SECRET: &str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";

    /// The cache-salt secret of the configurations below: `SECRET` in a file of mode 600, which each test process
    /// writes anew.
    static SECRET_FILE: LazyLock<PathBuf> = LazyLock::new(|| {
        let path = std::env::temp_dir().join("flatshare-gw-config-test-cache-salt-secret");
        let staged = path.with_extension(std::process::id().to_string());
        std::fs::write(&staged, format!("{SECRET}\n")).unwrap();
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::rename(&staged, &path).unwrap();
        path
    });

    /// The metrics key of the configurations below, `metrics-key` in a file of mode 600.
    static METRICS_FILE: LazyLock<PathBuf> = LazyLock::new(|| {
        let path = std::env::temp_dir().join("flatshare-gw-config-test-metrics-key");
        let staged = path.with_extension(std::process::id().to_string());
        std::fs::write(&staged, "metrics-key\n").unwrap();
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::rename(&staged, &path).unwrap();
        path
    });

    /// The configuration lines that name a cache-salt secret file and the metrics key's file.
    fn secret_file(path: &Path) -> String {
        format!(
            "cache_salt_secret_file = \"{}\"\nmetrics_key_file = \"{}\"",
            path.display(),
            METRICS_FILE.display()
        )
    }

    /// A configuration with one TCP listener, plus top-level keys and tables.
    fn config(top: &str, tables: &str) -> Result<Config, String> {
        Config::from_toml(&format!(
            "upstream = \"http://127.0.0.1:18000\"\n{}\n{top}\n[[listen]]\ntcp = \"127.0.0.1:0\"\n{tables}",
            secret_file(&SECRET_FILE)
        ))
        .map_err(|e| e.to_string())
    }

    #[test]
    fn defaults_follow_the_spec() {
        let config = config("", &format!("[tenants.agents]\nkeys_sha256 = [\"{KEY}\"]")).unwrap();
        assert_eq!(config.upstream.to_string(), "http://127.0.0.1:18000");
        assert_eq!(config.upstream_idle_timeout, Duration::from_secs(4));
        let vllm = PrioritySignals {
            vllm_header: true,
            ..PrioritySignals::default()
        };
        assert_eq!(config.upstream_priority, vllm);
        assert_eq!(
            (config.max_body_bytes, config.stream_buffer_frames),
            (32 << 20, 64)
        );
        assert_eq!(
            [
                config.stall_timeout,
                config.first_byte_timeout,
                config.drain_timeout,
                config.background_drain
            ],
            [30, 600, 30, 5].map(Duration::from_secs)
        );
        assert_eq!(
            (config.event_dedup_capacity, config.event_dedup_ttl),
            (100_000, Duration::from_secs(600))
        );
        let tenant = &config.tenants[0];
        assert_eq!(
            (tenant.trusted, tenant.default_class, tenant.max_class),
            (false, SloClass::Interactive, SloClass::Interactive)
        );
        assert_eq!(tenant.max_in_flight, 512);
        assert_eq!(
            (
                config.max_in_flight,
                config.max_queued,
                config.max_buffered_body_bytes
            ),
            (1024, 256, 256 << 20)
        );
        assert_eq!(
            SloClass::ALL.map(|class| config.class(class).max_in_flight),
            [128, 256, 512]
        );
        assert_eq!(tenant.keys_sha256[0][..2], [0x2c, 0x26]);
    }

    #[test]
    fn mistakes_are_rejected_with_their_cause() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        for (top, tables, cause) in [
            ("", "", "no [tenants"),
            ("", "[tenants.a]", "no API key and no listener"),
            ("", "[tenants.a]\nkeys_sha256 = [\"abc\"]", "64 hex"),
            (
                "",
                &format!("{a}\n[tenants.b]\nkeys_sha256 = [\"{KEY}\"]"),
                "listed twice",
            ),
            (
                "",
                &format!("{a}\ndefault_class = \"realtime\"\nmax_class = \"interactive\""),
                "stricter than",
            ),
            ("stall_timeout = 5", &a, "unknown field"),
            (
                "upstream_priority = [\"body\", \"body\"]",
                &a,
                "lists body twice",
            ),
            ("upstream_priority = [\"sglang\"]", &a, "unknown variant"),
            (
                "stall_timeout_s = 0",
                &a,
                "stall_timeout_s must be a positive",
            ),
            (
                "background_drain_s = 31",
                &a,
                "background_drain_s must be a number of seconds from 0 to drain_timeout_s (30)",
            ),
            (
                "drain_timeout_s = 2\nbackground_drain_s = 3",
                &a,
                "from 0 to drain_timeout_s (2)",
            ),
            ("background_drain_s = -1", &a, "from 0 to drain_timeout_s"),
            (
                "",
                "[[listen]]\ntcp = \"127.0.0.1:1\"\ntenant = \"a\"\n[tenants.a]",
                "TCP listener cannot",
            ),
            (
                "",
                "[[listen]]\nunix = \"relative.sock\"\ntenant = \"a\"\n[tenants.a]",
                "not absolute",
            ),
        ] {
            let err = config(top, tables).unwrap_err();
            assert!(err.contains(cause), "{err}");
        }
        let https = format!(
            "upstream = \"https://engine:8000/v1\"\n[[listen]]\ntcp = \"127.0.0.1:0\"\n{a}"
        );
        let err = Config::from_toml(&https).unwrap_err().to_string();
        assert!(err.contains("http://host:port"), "{err}");
    }

    #[test]
    fn a_deployment_may_cap_each_classs_replies() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let caps =
            |config: &Config| SloClass::ALL.map(|class| config.class(class).max_reply_tokens);
        assert_eq!(caps(&config("", &a).unwrap()), [None; 3]);
        // realtime_max_tokens is the realtime class's cap, as it was before the classes' tables.
        let capped = config("realtime_max_tokens = 64", &a).unwrap();
        assert_eq!(caps(&capped), [Some(64), None, None]);
        let tables = format!(
            "{a}\n[classes.realtime]\nmax_reply_tokens = 48\n[classes.background]\nmax_reply_tokens = 4096"
        );
        assert_eq!(
            caps(&config("", &tables).unwrap()),
            [Some(48), None, Some(4096)]
        );
        for (top, tables, cause) in [
            (
                "realtime_max_tokens = 0",
                a.clone(),
                "realtime_max_tokens must be positive",
            ),
            (
                "realtime_max_tokens = 64",
                format!("{a}\n[classes.realtime]\nmax_reply_tokens = 48"),
                "realtime_max_tokens and classes.realtime.max_reply_tokens are the same cap",
            ),
            (
                "",
                format!("{a}\n[classes.interactive]\nmax_reply_tokens = 0"),
                "classes.interactive.max_reply_tokens must be positive",
            ),
            (
                "",
                format!("{a}\n[classes.urgent]\nmax_reply_tokens = 1"),
                "unknown field `urgent`",
            ),
            (
                "",
                format!("{a}\n[classes.realtime]\nmax_tokens = 1"),
                "unknown field `max_tokens`",
            ),
        ] {
            let err = config(top, &tables).unwrap_err();
            assert!(err.contains(cause), "{err}");
        }
    }

    #[test]
    fn the_intakes_bounds_are_positive_and_hold_a_body_of_the_largest_size() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]\nmax_in_flight = 3");
        let tables = format!("{a}\n[classes.background]\nmax_in_flight = 2");
        let bounded = config("max_in_flight = 9\nmax_queued = 4", &tables).unwrap();
        assert_eq!((bounded.max_in_flight, bounded.max_queued), (9, 4));
        assert_eq!(bounded.tenants[0].max_in_flight, 3);
        assert_eq!(
            SloClass::ALL.map(|class| bounded.class(class).max_in_flight),
            [128, 256, 2]
        );
        for (top, tables, cause) in [
            (
                "max_in_flight = 0",
                a.clone(),
                "max_in_flight must be positive",
            ),
            ("max_queued = 0", a.clone(), "max_queued must be positive"),
            (
                "max_body_bytes = 1000\nmax_buffered_body_bytes = 999",
                a.clone(),
                "max_buffered_body_bytes (999) must be at least max_body_bytes (1000)",
            ),
            (
                "",
                format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]\nmax_in_flight = 0"),
                "tenants.a.max_in_flight must be positive",
            ),
            (
                "",
                format!("{a}\n[classes.realtime]\nmax_in_flight = 0"),
                "classes.realtime.max_in_flight must be positive",
            ),
        ] {
            let err = config(top, &tables).unwrap_err();
            assert!(err.contains(cause), "{err}");
        }
    }

    #[test]
    fn each_class_has_its_deadlines() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let config_of = |top: &str, tables: &str| config(top, &format!("{a}\n{tables}"));
        let defaults = config_of("", "").unwrap();
        let s = Duration::from_secs;
        let deadlines = |config: &Config, class| {
            let c = config.class(class);
            [
                c.upload_timeout,
                c.queue_timeout,
                c.headers_timeout,
                c.first_content_timeout,
                c.idle_timeout,
                c.total_timeout,
            ]
        };
        assert_eq!(
            deadlines(&defaults, SloClass::Realtime),
            [s(60), s(120), s(600), s(60), s(30), s(300)]
        );
        assert_eq!(
            deadlines(&defaults, SloClass::Interactive),
            [s(60), s(120), s(600), s(600), s(600), s(1800)]
        );
        assert_eq!(
            deadlines(&defaults, SloClass::Background),
            [s(60), s(120), s(600), s(600), s(600), s(3600)]
        );
        // first_byte_timeout_s is every class's headers_timeout_s unless a class sets its own.
        let set = config_of(
            "first_byte_timeout_s = 90",
            "[classes.background]\nheaders_timeout_s = 30\nidle_timeout_s = 2.5\ntotal_timeout_s = 7200",
        )
        .unwrap();
        assert_eq!(set.class(SloClass::Realtime).headers_timeout, s(90));
        assert_eq!(
            deadlines(&set, SloClass::Background),
            [
                s(60),
                s(120),
                s(30),
                s(600),
                Duration::from_millis(2500),
                s(7200)
            ]
        );
        for (tables, cause) in [
            (
                "[classes.realtime]\nidle_timeout_s = 0",
                "classes.realtime.idle_timeout_s must be a positive number of seconds",
            ),
            (
                "[classes.interactive]\nupload_timeout_s = -1",
                "classes.interactive.upload_timeout_s must be a positive",
            ),
            (
                "[classes.background]\nfirst_byte_timeout_s = 1",
                "unknown field",
            ),
        ] {
            let err = config_of("", tables).unwrap_err();
            assert!(err.contains(cause), "{err}");
        }
    }

    #[test]
    fn the_canary_is_on_unless_switched_off_and_realtime_has_an_objective() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let defaults = config("", &a).unwrap();
        let canary = Canary {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(60),
            degraded_after: 3,
        };
        assert_eq!(defaults.canary, Some(canary));
        assert_eq!(
            SloClass::ALL.map(|class| defaults.class(class).first_content_objective),
            [Some(Duration::from_millis(250)), None, None]
        );
        let set = config(
            "",
            &format!(
                "{a}\n[canary]\ninterval_s = 5\ndegraded_after = 1\n[classes.interactive]\nfirst_content_objective_s = 2"
            ),
        )
        .unwrap();
        assert_eq!(
            set.canary,
            Some(Canary {
                interval: Duration::from_secs(5),
                degraded_after: 1,
                ..canary
            })
        );
        assert_eq!(
            set.class(SloClass::Interactive).first_content_objective,
            Some(Duration::from_secs(2))
        );
        // Off, its other keys may stay, as E2's runs merge enabled = false over the deployment's table.
        let off = config(
            "",
            &format!("{a}\n[canary]\nenabled = false\ninterval_s = 5"),
        )
        .unwrap();
        assert_eq!(off.canary, None);
        for (tables, cause) in [
            (
                "[canary]\ndegraded_after = 0",
                "canary.degraded_after must be positive",
            ),
            (
                "[canary]\ntimeout_s = 0",
                "canary.timeout_s must be a positive",
            ),
            (
                "[canary]\nroute = \"/v1/completions\"",
                "unknown field `route`",
            ),
            (
                "[classes.realtime]\nfirst_content_objective_s = 0",
                "classes.realtime.first_content_objective_s must be a positive",
            ),
        ] {
            let err = config("", &format!("{a}\n{tables}")).unwrap_err();
            assert!(err.contains(cause), "{err}");
        }
    }

    /// A configuration of `[[nodes]]` in place of `upstream`, with `top`'s keys and `nodes` after the tenants.
    fn nodes_config(top: &str, nodes: &str) -> Result<Config, String> {
        Config::from_toml(&format!(
            "{}\n{top}\n[[listen]]\ntcp = \"127.0.0.1:0\"\n[tenants.a]\nkeys_sha256 = [\"{KEY}\"]\n{nodes}",
            secret_file(&SECRET_FILE)
        ))
        .map_err(|e| e.to_string())
    }

    const TWO_NODES: &str = r#"
[[nodes]]
name = "phastform"
upstream = "unix:/run/user/1007/flatshare/phastform-engine.sock"
stt = "unix:/run/user/1007/flatshare/phastform-stt.sock"
tts = "unix:/run/user/1007/flatshare/phastform-tts.sock"

[[nodes]]
name = "gray"
upstream = "unix:/run/user/1007/flatshare/gray-engine.sock"
roles = ["background"]
max_calls = 0
"#;

    #[test]
    fn nodes_take_both_roles_two_calls_and_the_placements_defaults() {
        let two = nodes_config("", TWO_NODES).unwrap();
        let [phastform, gray] = &two.nodes[..] else {
            panic!("{:?}", two.nodes)
        };
        assert_eq!(phastform.name, "phastform");
        assert_eq!(
            (
                phastform.realtime,
                phastform.background,
                phastform.max_calls
            ),
            (true, true, 2)
        );
        assert_eq!(
            phastform.stt,
            Some(UpstreamAddr::Unix(
                "/run/user/1007/flatshare/phastform-stt.sock".into()
            ))
        );
        // A background node needs no call slot and no speech servers.
        assert_eq!(
            (gray.realtime, gray.background, &gray.tts),
            (false, true, &None)
        );
        // The first node's engine stands for the one upstream (the start line, the default node).
        assert_eq!(two.upstream, phastform.upstream);
        assert_eq!(two.placement, Placement::default());
        assert_eq!(
            two.placement,
            Placement {
                idle: Duration::from_secs(300),
                health_interval: Duration::from_secs(1),
                health_timeout: Duration::from_secs(1),
                unhealthy_after: 3,
                healthy_after: 2,
                failover_background_max_in_flight: 0,
            }
        );
        let tuned = format!(
            "{TWO_NODES}[placement]\nidle_s = 60\nhealth_interval_s = 0.5\nunhealthy_after = 5\n\
             failover_background_max_in_flight = 2\n"
        );
        let placement = nodes_config("", &tuned).unwrap().placement;
        assert_eq!(
            (
                placement.idle,
                placement.health_interval,
                placement.unhealthy_after,
                placement.failover_background_max_in_flight
            ),
            (Duration::from_secs(60), Duration::from_millis(500), 5, 2)
        );
        // Without [[nodes]], none, and the placement's defaults.
        let one = config("", &format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]")).unwrap();
        assert!(one.nodes.is_empty());
    }

    #[test]
    fn nodes_are_refused_with_an_upstream_an_admission_policy_or_a_role_no_node_takes() {
        let node = |name: &str, keys: &str| {
            format!("[[nodes]]\nname = \"{name}\"\nupstream = \"unix:/run/{name}.sock\"\n{keys}\n")
        };
        let both = format!("{}{}", node("a", ""), node("b", ""));
        for (top, nodes, cause) in [
            (
                "upstream = \"http://127.0.0.1:18000\"",
                both.clone(),
                "set upstream or [[nodes]], not both",
            ),
            (
                "",
                format!("{both}[admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 1"),
                "an [admission] policy is for one upstream",
            ),
            (
                "",
                format!("{}{}", node("a", ""), node("a", "")),
                "nodes: \"a\" is named twice",
            ),
            (
                "",
                format!(
                    "{}[[nodes]]\nname = \"b\"\nupstream = \"unix:/run/a.sock\"\n",
                    node("a", "")
                ),
                "nodes.b.upstream is nodes.a's",
            ),
            ("", node("a b", ""), "nodes[0].name \"a b\""),
            ("", node("a", "roles = []"), "nodes.a.roles is empty"),
            (
                "",
                node("a", "roles = [\"realtime\", \"realtime\"]"),
                "nodes.a.roles names a role twice",
            ),
            (
                "",
                node("a", "roles = [\"realtime\"]"),
                "no node has the background role",
            ),
            (
                "",
                node("a", "max_calls = 0"),
                "nodes.a.max_calls must be positive",
            ),
            (
                "",
                node("a", "tts = \"http://127.0.0.1:9000/tts\""),
                "nodes.a.tts \"http://127.0.0.1:9000/tts\": expected",
            ),
            (
                "",
                node("a", "speech = \"unix:/run/a-stt.sock\""),
                "unknown field `speech`",
            ),
            (
                "",
                format!("{both}[placement]\nunhealthy_after = 0"),
                "placement.unhealthy_after must be positive",
            ),
            (
                "",
                format!("{both}[placement]\nidle_s = 0"),
                "placement.idle_s must be a positive",
            ),
        ] {
            let err = nodes_config(top, &nodes).unwrap_err();
            assert!(err.contains(cause), "{cause}: {err}");
        }
        // [placement] without [[nodes]].
        let err = config(
            "",
            &format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]\n[placement]\nidle_s = 1"),
        )
        .unwrap_err();
        assert!(err.contains("[placement] needs [[nodes]]"), "{err}");
    }

    #[test]
    fn a_node_with_the_background_role_may_bound_its_agents_sessions() {
        let bounded = TWO_NODES.replace(
            "phastform-tts.sock\"\n",
            "phastform-tts.sock\"\nmax_agents = 4\n",
        );
        let two = nodes_config("", &bounded).unwrap();
        let [phastform, gray] = &two.nodes[..] else {
            panic!("{:?}", two.nodes)
        };
        // Without the key, no bound.
        assert_eq!((phastform.max_agents, gray.max_agents), (Some(4), None));
        let node = |name: &str, keys: &str| {
            format!("[[nodes]]\nname = \"{name}\"\nupstream = \"unix:/run/{name}.sock\"\n{keys}\n")
        };
        let calls_only = format!(
            "{}{}",
            node("a", "roles = [\"realtime\"]\nmax_agents = 2"),
            node("b", "roles = [\"background\"]")
        );
        for (nodes, cause) in [
            (
                node("a", "max_agents = 0"),
                "nodes.a.max_agents must be positive, or left out for no bound",
            ),
            (
                calls_only,
                "nodes.a.max_agents is for a node with the background role",
            ),
        ] {
            let err = nodes_config("", &nodes).unwrap_err();
            assert!(err.contains(cause), "{cause}: {err}");
        }
    }

    #[test]
    fn requests_ask_for_one_choice_unless_the_deployment_allows_more() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        assert_eq!(config("", &a).unwrap().max_choices, 1);
        assert_eq!(config("max_choices = 4", &a).unwrap().max_choices, 4);
        let err = config("max_choices = 0", &a).unwrap_err();
        assert!(err.contains("max_choices must be positive"), "{err}");
    }

    #[test]
    fn background_requests_may_be_cut_from_the_drains_start_to_its_end() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let at_once = config("background_drain_s = 0", &a).unwrap();
        assert_eq!(at_once.background_drain, Duration::ZERO);
        let whole = config("drain_timeout_s = 10\nbackground_drain_s = 10", &a).unwrap();
        assert_eq!(whole.background_drain, whole.drain_timeout);
    }

    #[test]
    fn an_upstream_is_a_tcp_address_or_a_unix_socket() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let upstream = |value: &str| {
            Config::from_toml(&format!(
                "upstream = \"{value}\"\n{}\n[[listen]]\ntcp = \"127.0.0.1:0\"\n{a}",
                secret_file(&SECRET_FILE)
            ))
            .map(|config| config.upstream)
            .map_err(|e| e.to_string())
        };
        let socket = "/run/user/1007/flatshare/phastform-engine.sock";
        assert_eq!(
            upstream(&format!("unix:{socket}")).unwrap(),
            UpstreamAddr::Unix(socket.into())
        );
        assert_eq!(
            upstream("http://127.0.0.1:18000").unwrap(),
            UpstreamAddr::Tcp("127.0.0.1:18000".parse().unwrap())
        );
        // The start line and the errors name it as it was written.
        assert_eq!(
            UpstreamAddr::Unix(socket.into()).to_string(),
            format!("unix:{socket}")
        );
        let long = format!("unix:/{}", "s".repeat(200));
        for (value, cause) in [
            ("unix:relative.sock", "or unix: and an absolute socket path"),
            ("unix:", "or unix: and an absolute socket path"),
            (
                "unix://engine/x.sock",
                "or unix: and an absolute socket path",
            ),
            (&long, "the path is too long for a Unix socket address"),
            (
                "http+unix://%2Frun%2Fx.sock",
                "or unix: and an absolute socket path",
            ),
        ] {
            let err = upstream(value).unwrap_err();
            assert!(
                err.contains(&format!("upstream {value:?}: ")) && err.contains(cause),
                "{err}"
            );
        }
    }

    #[test]
    fn the_cache_salt_secret_is_a_private_file_of_32_bytes() {
        let dir = std::env::temp_dir().join(format!("flatshare-gw-secret-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = |name: &str, text: &str, mode: u32| {
            let path = dir.join(name);
            std::fs::write(&path, text).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            path
        };
        let with = |secret: &str| {
            Config::from_toml(&format!(
                "upstream = \"http://127.0.0.1:18000\"\n{secret}\n[[listen]]\ntcp = \"127.0.0.1:0\"\n\
                 [tenants.a]\nkeys_sha256 = [\"{KEY}\"]"
            ))
            .map_err(|e| e.to_string())
        };
        let private = file("private", &format!("{SECRET}\n"), 0o600);
        let config = with(&secret_file(&private)).unwrap();
        let bytes = CacheSaltSecret(std::array::from_fn(|i| i as u8));
        assert_eq!(config.cache_salt_secret, bytes);
        assert_eq!(
            format!("{bytes:?}"),
            "CacheSaltSecret(..)",
            "it is never printed"
        );

        let unreadable = dir.join("a directory");
        std::fs::create_dir(&unreadable).unwrap();
        std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o700)).unwrap();
        for (path, cause) in [
            (
                file("readable", SECRET, 0o644),
                "mode 644 lets other users open it; make it 600",
            ),
            (file("group", SECRET, 0o640), "mode 640 lets other users"),
            (
                file("short", &SECRET[2..], 0o600),
                "expected 64 hex characters",
            ),
            (dir.join("missing"), "No such file or directory"),
            (unreadable, "Is a directory"),
            (PathBuf::from("relative/secret"), "not an absolute path"),
        ] {
            let err = with(&secret_file(&path)).unwrap_err();
            assert!(
                err.starts_with(&format!("cache_salt_secret_file {}: ", path.display()))
                    && err.contains(cause),
                "{err}"
            );
        }
        let err = with("").unwrap_err();
        assert!(err.starts_with("no cache_salt_secret_file"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tcp_listener_needs_a_metrics_key_in_a_private_file() {
        let salt = format!("cache_salt_secret_file = \"{}\"", SECRET_FILE.display());
        let with = |listen: &str, key: &str| {
            Config::from_toml(&format!(
                "upstream = \"http://127.0.0.1:18000\"\n{salt}\n{key}\n[[listen]]\n{listen}\n\
                 [tenants.a]\nkeys_sha256 = [\"{KEY}\"]\ntrusted = true"
            ))
            .map_err(|e| e.to_string())
        };
        let tcp = "tcp = \"127.0.0.1:0\"";
        let unix = "unix = \"/run/user/1007/flatshare/gateway.sock\"\ntenant = \"a\"";
        let named = |path: &Path| format!("metrics_key_file = \"{}\"", path.display());

        let config = with(tcp, &named(&METRICS_FILE)).unwrap();
        // SHA-256 of "metrics-key": the key itself is never kept, and never printed.
        let digest = Sha256::digest("metrics-key").into();
        assert_eq!(config.metrics_key, Some(MetricsKey(digest)));
        assert_eq!(
            format!("{:?}", config.metrics_key.unwrap()),
            "MetricsKey(..)"
        );
        // Only a TCP listener needs it: a Unix socket's file mode keeps other users out.
        assert_eq!(with(unix, "").unwrap().metrics_key, None);
        let err = with(tcp, "").unwrap_err();
        assert!(
            err.starts_with("no metrics_key_file: a TCP listener serves /metrics"),
            "{err}"
        );

        let dir =
            std::env::temp_dir().join(format!("flatshare-gw-metrics-key-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = |name: &str, text: &str, mode: u32| {
            let path = dir.join(name);
            std::fs::write(&path, text).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            path
        };
        for (path, cause) in [
            (
                file("readable", "k\n", 0o644),
                "mode 644 lets other users open it",
            ),
            (file("empty", "\n", 0o600), "expected one key on one line"),
            (
                file("two", "k1 k2\n", 0o600),
                "expected one key on one line",
            ),
            (dir.join("missing"), "No such file or directory"),
            (PathBuf::from("relative/key"), "not an absolute path"),
        ] {
            let err = with(tcp, &named(&path)).unwrap_err();
            assert!(
                err.starts_with(&format!("metrics_key_file {}: ", path.display()))
                    && err.contains(cause),
                "{err}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tcp_listener_binds_loopback() {
        // The review's probe P1: beyond loopback, bearer keys would cross the network in cleartext, and the
        // keyless /metrics, /health and /ready would answer it.
        let listen = |tcp: &str| {
            Config::from_toml(&format!(
                "upstream = \"http://127.0.0.1:18000\"\n{}\n[[listen]]\ntcp = \"{tcp}\"\n\
                 [tenants.a]\nkeys_sha256 = [\"{KEY}\"]",
                secret_file(&SECRET_FILE)
            ))
            .map_err(|e| e.to_string())
        };
        for tcp in ["0.0.0.0:18080", "[::]:18080", "192.0.2.7:18080"] {
            let err = listen(tcp).unwrap_err();
            assert!(
                err.starts_with(&format!("listen[0]: tcp {tcp} is not a loopback address")),
                "{err}"
            );
        }
        for tcp in ["127.0.0.1:18080", "127.0.0.2:18080", "[::1]:18080"] {
            let config = listen(tcp).unwrap();
            assert_eq!(
                config.listeners[0].addr,
                ListenAddr::Tcp(tcp.parse().unwrap())
            );
        }
    }

    #[test]
    fn upstream_priority_is_a_set_of_signals() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let none = config("upstream_priority = []", &a).unwrap();
        assert_eq!(none.upstream_priority, PrioritySignals::default());
        let all = config(
            "upstream_priority = [\"smg-header\", \"body\", \"vllm-header\"]",
            &a,
        )
        .unwrap();
        assert_eq!(
            all.upstream_priority,
            PrioritySignals {
                vllm_header: true,
                body: true,
                smg_header: true
            }
        );
    }

    const DEPLOY_EXAMPLE: &str = include_str!("../../../deploy/cyan/gateway.example.toml");

    /// The deploy example with its placeholders filled in.
    fn deploy_example() -> String {
        // SHA-256 of "bar".
        let other = "fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9";
        DEPLOY_EXAMPLE
            .replace("<sha256 of the agents key>", KEY)
            .replace("<sha256 of the bench key>", other)
            .replace(
                "<home>/.config/flatshare/cache-salt-secret",
                &SECRET_FILE.display().to_string(),
            )
            .replace(
                "<home>/.config/flatshare/metrics-key",
                &METRICS_FILE.display().to_string(),
            )
    }

    #[test]
    fn the_deploy_example_is_valid_once_its_placeholders_are_filled_in() {
        let err = Config::from_toml(DEPLOY_EXAMPLE).unwrap_err().to_string();
        assert!(err.contains("64 hex"), "placeholders are refused: {err}");

        let config = Config::from_toml(&deploy_example()).unwrap();
        let unix = ListenAddr::Unix("/run/user/1007/flatshare/gateway.sock".into());
        assert_eq!(config.listeners[1].addr, unix);
        assert_eq!(config.listeners[1].tenant.as_deref(), Some("voice"));
        let classes: Vec<_> = config
            .tenants
            .iter()
            .map(|t| (t.name.as_str(), t.trusted, t.default_class, t.max_class))
            .collect();
        use SloClass::*;
        assert_eq!(
            classes,
            [
                ("agents", true, Background, Interactive),
                ("bench", false, Background, Background),
                ("voice", true, Realtime, Realtime),
            ]
        );
    }

    /// The review's probe P5: every committed arm file's `[gateway]` table, merged into the deploy example as
    /// `flatshare-bench e2 run` merges it into cyan's gateway.toml, is a configuration this build takes. A key renamed
    /// here would otherwise show only when the gateway failed to start on cyan.
    #[test]
    fn every_arm_files_gateway_table_is_valid_over_the_deploy_example() {
        // Tables merge key by key, and any other value replaces.
        fn merge(base: &mut toml::Table, overlay: &toml::Table) {
            for (key, value) in overlay {
                match (base.get_mut(key), value) {
                    (Some(toml::Value::Table(base)), toml::Value::Table(overlay)) => {
                        merge(base, overlay)
                    }
                    _ => {
                        base.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        let arms = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/e2/arms");
        let mut checked = 0;
        for entry in std::fs::read_dir(&arms).unwrap() {
            let path = entry.unwrap().path();
            let arm: toml::Table = std::fs::read_to_string(&path).unwrap().parse().unwrap();
            let Some(toml::Value::Table(overlay)) = arm.get("gateway") else {
                continue;
            };
            let mut config: toml::Table = deploy_example().parse().unwrap();
            merge(&mut config, overlay);
            if let Err(err) = Config::from_toml(&toml::to_string(&config).unwrap()) {
                panic!("{}: {err}", path.display());
            }
            checked += 1;
        }
        assert!(checked > 0, "no arm file has a [gateway] table");
    }
}
