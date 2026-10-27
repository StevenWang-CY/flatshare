//! What the gateway removes from a client request, what it resolves from it (session and class), and what it
//! sets for the engine. The names come from `flatshare_proto::headers`.

use std::fmt;
use std::io::Write as _;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use flatshare_proto::headers;
use flatshare_proto::{SessionId, SloClass, UnknownClass};
use http::header::{self, HeaderMap, HeaderName, HeaderValue};
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

use crate::auth::Tenant;
use crate::bounds;
use crate::config::PrioritySignals;
use crate::metrics::{Metrics, Route};

pub(crate) static SESSION_ID: HeaderName = HeaderName::from_static(headers::SESSION_ID);
pub(crate) static VLLM_PRIORITY: HeaderName = HeaderName::from_static(headers::VLLM_PRIORITY);
pub(crate) static SMG_PRIORITY: HeaderName = HeaderName::from_static(headers::SMG_PRIORITY);
pub(crate) static REQUEST_ID: HeaderName = HeaderName::from_static(headers::REQUEST_ID);
pub(crate) static RESOLVED_SESSION_ID: HeaderName =
    HeaderName::from_static(headers::RESOLVED_SESSION_ID);

/// The `x-flatshare-` headers counted under their own name when removed.
const NAMED_FLATSHARE_HEADERS: [&str; 2] = [headers::CLASS, headers::RESOLVED_SESSION_ID];

/// Metric label for removed `x-flatshare-` headers that `flatshare_proto` does not name, which keeps the label bounded.
const OTHER_FLATSHARE_HEADER: &str = "x-flatshare-*";

/// Every `field` label of `flatshare_gateway_sanitized_total`.
pub(crate) fn sanitized_fields() -> impl Iterator<Item = &'static str> {
    headers::ENGINE_CONTROL_HEADERS
        .into_iter()
        .chain([headers::REQUEST_ID])
        .chain(NAMED_FLATSHARE_HEADERS)
        .chain([OTHER_FLATSHARE_HEADER])
        .chain(headers::ENGINE_CONTROL_BODY_FIELDS)
}

/// Hop-by-hop headers (RFC 9110 section 7.6.1, plus the legacy `Keep-Alive` and `Proxy-Connection`).
static HOP_BY_HOP: [HeaderName; 8] = [
    header::CONNECTION,
    HeaderName::from_static("keep-alive"),
    HeaderName::from_static("proxy-connection"),
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
];

/// Request headers that do not apply to the upstream request: the client's key, which only the gateway checks;
/// the gateway's host; the length and type of a body the gateway rewrites; and `Expect`, which the gateway
/// already answered by reading the body.
static NOT_FORWARDED: [HeaderName; 5] = [
    header::AUTHORIZATION,
    header::HOST,
    header::CONTENT_LENGTH,
    header::CONTENT_TYPE,
    header::EXPECT,
];

/// The values a request carries for the gateway itself.
#[derive(Debug, Default)]
pub(crate) struct ClientInputs {
    /// `X-Session-ID`, kept only from a trusted tenant.
    pub session_id: Option<HeaderValue>,
    pub claude_code_session_id: Option<HeaderValue>,
    /// `X-Flatshare-Class`, kept only from a trusted tenant.
    pub class: Option<HeaderValue>,
}

/// Removes from `headers` everything the engine must not receive from a client, counting each input a client
/// set but may not control, and returns the inputs the gateway consumes.
pub(crate) fn sanitize_headers(
    headers: &mut HeaderMap,
    trusted: bool,
    metrics: &Metrics,
) -> ClientInputs {
    let listed = connection_listed(headers);
    let mut inputs = ClientInputs {
        claude_code_session_id: headers.remove(headers::CLAUDE_CODE_SESSION_ID),
        ..ClientInputs::default()
    };
    if trusted {
        inputs.session_id = headers.remove(&SESSION_ID);
        inputs.class = headers.remove(headers::CLASS);
    }
    for name in headers::ENGINE_CONTROL_HEADERS
        .into_iter()
        .chain([headers::REQUEST_ID])
    {
        if headers.remove(name).is_some() {
            metrics.sanitized(name);
        }
    }
    let flatshare_headers: Vec<HeaderName> = headers
        .keys()
        .filter(|name| name.as_str().starts_with(headers::FLATSHARE_PREFIX))
        .cloned()
        .collect();
    for name in flatshare_headers {
        headers.remove(&name);
        metrics.sanitized(
            NAMED_FLATSHARE_HEADERS
                .into_iter()
                .find(|known| *known == name.as_str())
                .unwrap_or(OTHER_FLATSHARE_HEADER),
        );
    }
    for name in HOP_BY_HOP.iter().chain(&NOT_FORWARDED).chain(&listed) {
        headers.remove(name);
    }
    inputs
}

/// Removes hop-by-hop headers from an upstream response before it is relayed.
pub(crate) fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let listed = connection_listed(headers);
    for name in HOP_BY_HOP.iter().chain(&listed) {
        headers.remove(name);
    }
}

/// The header names a `Connection` header lists, which are hop-by-hop as well.
fn connection_listed(headers: &HeaderMap) -> Vec<HeaderName> {
    headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect()
}

/// The request's class: `X-Flatshare-Class` from a trusted tenant, else the tenant's default, clamped to the tenant's
/// strictest allowed class. Returns whether it was clamped.
pub(crate) fn resolve_class(
    tenant: &Tenant,
    requested: Option<&HeaderValue>,
) -> Result<(SloClass, bool), UnknownClass> {
    let class = match requested {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| UnknownClass(String::from_utf8_lossy(value.as_bytes()).into_owned()))?,
        None => tenant.default_class,
    };
    // Classes order from strictest to most lenient, so "stricter than allowed" is "less than".
    Ok(if class < tenant.max_class {
        (tenant.max_class, true)
    } else {
        (class, false)
    })
}

/// The session of a request, from the first present source in PROTOCOL.md's order: a trusted `X-Session-ID`,
/// `X-Claude-Code-Session-ID`, the body's `prompt_cache_key`. A trusted tenant's valid session id is used as it
/// is. Any other source is hashed into one, scoped to the tenant: every source of an untrusted tenant, so that it
/// cannot name another tenant's session, and a source that is not a valid session id. With no source the gateway
/// mints one.
pub(crate) fn resolve_session(
    tenant: &Tenant,
    inputs: &ClientInputs,
    prompt_cache_key: Option<&str>,
) -> SessionId {
    match session_source(inputs, prompt_cache_key) {
        Some(raw) => std::str::from_utf8(raw)
            .ok()
            .filter(|_| tenant.trusted)
            .and_then(|id| SessionId::new(id).ok())
            .unwrap_or_else(|| hashed_session(&tenant.name, raw)),
        None => new_id("m-", &random_hex()),
    }
}

/// Whether a request names its session (any of its sources is present), rather than having the gateway mint one,
/// which no later request can name.
pub(crate) fn names_session(inputs: &ClientInputs, prompt_cache_key: Option<&str>) -> bool {
    session_source(inputs, prompt_cache_key).is_some()
}

/// The first present source of a request's session, in PROTOCOL.md's order.
fn session_source<'a>(
    inputs: &'a ClientInputs,
    prompt_cache_key: Option<&'a str>,
) -> Option<&'a [u8]> {
    let sources = [
        inputs.session_id.as_ref().map(HeaderValue::as_bytes),
        inputs
            .claude_code_session_id
            .as_ref()
            .map(HeaderValue::as_bytes),
        prompt_cache_key.map(str::as_bytes),
    ];
    sources.into_iter().flatten().find(|raw| !raw.is_empty())
}

fn hashed_session(tenant: &str, raw: &[u8]) -> SessionId {
    let digest = Sha256::new()
        .chain_update(tenant)
        .chain_update([0])
        .chain_update(raw)
        .finalize();
    new_id("h-", &hex(&digest[..16]))
}

fn new_id(prefix: &str, hex: &str) -> SessionId {
    SessionId::new(format!("{prefix}{hex}"))
        .expect("a prefix and hex digits form a valid session id")
}

/// Sets the engine inputs the gateway owns: the session, the request id, and the class's priority in the
/// headers `signals` names.
pub(crate) fn set_engine_headers(
    headers: &mut HeaderMap,
    session: &SessionId,
    class: SloClass,
    request_id: &HeaderValue,
    signals: PrioritySignals,
) {
    headers.insert(SESSION_ID.clone(), session_header(session));
    set_priority(headers, class, signals);
    headers.insert(REQUEST_ID.clone(), request_id.clone());
}

/// Sets the class's priority in the headers `signals` names: the request's, and the canary's.
pub(crate) fn set_priority(headers: &mut HeaderMap, class: SloClass, signals: PrioritySignals) {
    if signals.vllm_header {
        headers.insert(
            VLLM_PRIORITY.clone(),
            HeaderValue::from(class.engine_priority()),
        );
    }
    if signals.smg_header {
        headers.insert(
            SMG_PRIORITY.clone(),
            HeaderValue::from_static(smg_class(class)),
        );
    }
}

/// The SMG class of an SLO class. SMG's fourth class, `default`, is what it gives a request without one.
fn smg_class(class: SloClass) -> &'static str {
    match class {
        SloClass::Realtime => "system",
        SloClass::Interactive => "interactive",
        SloClass::Background => "bulk",
    }
}

pub(crate) fn session_header(session: &SessionId) -> HeaderValue {
    HeaderValue::from_str(session.as_str()).expect("session ids are header-safe")
}

/// A request body after sanitization.
#[derive(Debug)]
pub(crate) struct Body {
    pub json: Bytes,
    pub prompt_cache_key: Option<String>,
}

/// What a rewrite bounds ([`crate::bounds`]): the reply of a request of `route`, at `cap` tokens if its class has
/// one, and its choices, at `max_choices`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub route: Route,
    pub cap: Option<u32>,
    pub max_choices: u32,
}

/// Why a request body was rejected (400).
#[derive(Debug)]
pub(crate) enum InvalidBody {
    /// Not a JSON object.
    Json(serde_json::Error),
    /// A reply limit or a choice count the gateway does not forward ([`crate::bounds`]).
    Bounds(bounds::Invalid),
}

impl fmt::Display for InvalidBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(err) => write!(f, "the request body must be a JSON object: {err}"),
            Self::Bounds(invalid) => invalid.fmt(f),
        }
    }
}

/// Removes the engine-control fields from a JSON object body and sets the tenant's `cache_salt`, and the
/// gateway's own `priority` if it is given one. Its reply limits and choices are checked ([`crate::bounds`]), and
/// with a cap, the reply is capped in the same pass, to the bytes [`cap_reply`] would make of the rewritten body:
/// the route's own limit fields first, at the lower of the client's limit and the cap, and the other routes' gone.
///
/// Only the top level is parsed: values stay raw slices of the input and are copied through unchanged, so the
/// cost is one validating scan and one copy whatever the body's size. Keys are compared after JSON unescaping,
/// so an escaped `"pri\u006frity"` is removed too, and an escaped `"max_\u0074okens"` is the `max_tokens` it
/// spells.
pub(crate) fn rewrite_body(
    body: &[u8],
    cache_salt: &str,
    priority: Option<i32>,
    bounds: Bounds,
    metrics: &Metrics,
) -> Result<Body, InvalidBody> {
    let TopLevel(fields) = serde_json::from_slice(body).map_err(InvalidBody::Json)?;
    let asked = bounds::read(
        fields.iter().map(|(key, value)| (key.as_str(), *value)),
        bounds.max_choices,
    )
    .map_err(InvalidBody::Bounds)?;
    let mut json = Vec::with_capacity(body.len() + 64);
    let mut prompt_cache_key = None;
    json.push(b'{');
    // No reply limit is an engine-control field, so the body the rewrite keeps has the limits this one has.
    let capped = bounds.cap.map(|cap| asked.capped(bounds.route, cap));
    if let Some(limit) = capped {
        bounds::write_limit(&mut json, bounds.route, limit);
    }
    for (key, value) in &fields {
        if let Some(field) = headers::ENGINE_CONTROL_BODY_FIELDS
            .into_iter()
            .find(|field| *field == key.as_str())
        {
            metrics.sanitized(field);
            continue;
        }
        if capped.is_some() && bounds::is_reply_limit(key) {
            continue;
        }
        if key == "prompt_cache_key" {
            prompt_cache_key = serde_json::from_str(value.get()).ok();
        }
        serde_json::to_writer(&mut json, key).expect("writing to a Vec cannot fail");
        json.push(b':');
        json.extend_from_slice(value.get().as_bytes());
        json.push(b',');
    }
    if let Some(priority) = priority {
        write!(json, "\"priority\":{priority},").expect("writing to a Vec cannot fail");
    }
    json.extend_from_slice(b"\"cache_salt\":");
    serde_json::to_writer(&mut json, cache_salt).expect("writing to a Vec cannot fail");
    json.push(b'}');
    Ok(Body {
        json: json.into(),
        prompt_cache_key,
    })
}

/// Caps a request's reply at `cap` tokens (degrade's) in a body [`rewrite_body`] wrote: the route's own limit
/// fields first, at the lower of the body's limit and `cap`, and the other routes' removed. Capping a capped body
/// again keeps the lower of the two caps, byte for byte as one cap at the lower would.
pub(crate) fn cap_reply(body: &[u8], route: Route, cap: u32) -> Bytes {
    let TopLevel(fields) =
        serde_json::from_slice(body).expect("the gateway writes bodies as JSON objects");
    // The rewrite checked the body's limits and choices, and writes no field twice.
    let asked = bounds::read(
        fields.iter().map(|(key, value)| (key.as_str(), *value)),
        u32::MAX,
    )
    .expect("a rewritten body's limits were checked");
    let mut json = Vec::with_capacity(body.len() + 32);
    json.push(b'{');
    bounds::write_limit(&mut json, route, asked.capped(route, cap));
    for (key, value) in &fields {
        if bounds::is_reply_limit(key) {
            continue;
        }
        serde_json::to_writer(&mut json, key).expect("writing to a Vec cannot fail");
        json.push(b':');
        json.extend_from_slice(value.get().as_bytes());
        json.push(b',');
    }
    // Every member ends with a comma, and a rewritten body has its cache salt among them.
    *json.last_mut().expect("a member was written") = b'}';
    json.into()
}

/// The top-level members of a JSON object, in order and duplicates kept, with raw values.
pub(crate) struct TopLevel<'a>(pub Vec<(String, &'a RawValue)>);

impl<'de> serde::Deserialize<'de> for TopLevel<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Members;
        impl<'de> Visitor<'de> for Members {
            type Value = TopLevel<'de>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut members = Vec::with_capacity(map.size_hint().unwrap_or(16));
                while let Some(member) = map.next_entry()? {
                    members.push(member);
                }
                Ok(TopLevel(members))
            }
        }
        deserializer.deserialize_map(Members)
    }
}

/// Lowercase hexadecimal.
pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|b| [DIGITS[usize::from(b >> 4)], DIGITS[usize::from(b & 15)]])
        .map(char::from)
        .collect()
}

/// 128 random bits as 32 hex digits, from the operating system's generator.
pub(crate) fn random_hex() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("the operating system's random number generator failed");
    hex(&bytes)
}

/// A request id: 32 hex digits, a random per-process prefix and a counter. Request ids must be unique, not
/// unpredictable (clients cannot set them), so this needs no system call per request.
pub(crate) fn request_id() -> HeaderValue {
    static PREFIX: LazyLock<u64> = LazyLock::new(|| {
        getrandom::u64().expect("the operating system's random number generator failed")
    });
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    HeaderValue::from_str(&format!("{:016x}{n:016x}", *PREFIX))
        .expect("hex digits are a valid header value")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn tenant(trusted: bool, default_class: SloClass, max_class: SloClass) -> Tenant {
        Tenant {
            name: "agents".into(),
            index: 0,
            max_in_flight: 1,
            trusted,
            default_class,
            max_class,
            cache_salt: "salt".into(),
        }
    }

    fn named(name: &str, trusted: bool) -> Tenant {
        Tenant {
            name: name.into(),
            ..tenant(trusted, SloClass::Interactive, SloClass::Interactive)
        }
    }

    fn header(value: &str) -> Option<HeaderValue> {
        Some(HeaderValue::from_str(value).unwrap())
    }

    #[test]
    fn sessions_resolve_in_protocol_order() {
        let agents = named("agents", true);
        let all = ClientInputs {
            session_id: header("voice-RM_a1B2c3"),
            claude_code_session_id: header("0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41"),
            class: None,
        };
        assert_eq!(
            resolve_session(&agents, &all, Some("pck")).as_str(),
            "voice-RM_a1B2c3"
        );
        let claude = ClientInputs {
            session_id: None,
            ..all
        };
        assert_eq!(
            resolve_session(&agents, &claude, Some("pck")).as_str(),
            "0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41"
        );
        assert_eq!(
            resolve_session(&agents, &ClientInputs::default(), Some("pck")).as_str(),
            "pck"
        );
        // An empty source is absent.
        let empty = ClientInputs {
            session_id: header(""),
            ..ClientInputs::default()
        };
        assert_eq!(
            resolve_session(&agents, &empty, Some("pck")).as_str(),
            "pck"
        );
    }

    #[test]
    fn invalid_sources_are_hashed_per_tenant_and_missing_ones_minted() {
        let (agents, inputs) = (named("agents", true), ClientInputs::default());
        let hashed = resolve_session(&agents, &inputs, Some("has spaces/and slashes"));
        // `printf 'agents\0has spaces/and slashes' | shasum -a 256 | cut -c1-32`
        assert_eq!(hashed.as_str(), "h-dc629ca91b9da396f87344e8a3e8ecbd");
        assert_ne!(
            resolve_session(
                &named("bench", true),
                &inputs,
                Some("has spaces/and slashes")
            ),
            hashed
        );
        let long = "a".repeat(SessionId::MAX_LEN + 1);
        assert!(
            resolve_session(&agents, &inputs, Some(&long))
                .as_str()
                .starts_with("h-")
        );

        let minted = resolve_session(&agents, &inputs, None);
        assert!(minted.as_str().starts_with("m-") && minted.as_str().len() == 34);
        assert!(minted.as_str()[2..].bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(minted, resolve_session(&agents, &inputs, None));
    }

    #[test]
    fn an_untrusted_tenant_names_only_sessions_of_its_own() {
        let bench = named("bench", false);
        let claude = ClientInputs {
            claude_code_session_id: header("voice-RM_victim"),
            ..ClientInputs::default()
        };
        // A valid session id from an untrusted tenant is hashed with the tenant all the same:
        // `printf 'bench\0voice-RM_victim' | shasum -a 256 | cut -c1-32`.
        assert_eq!(
            resolve_session(&bench, &claude, None).as_str(),
            "h-468e550b540d0075ae9e649823e0977e"
        );
        // `printf 'bench\0bench-7' | shasum -a 256 | cut -c1-32`
        assert_eq!(
            resolve_session(&bench, &ClientInputs::default(), Some("bench-7")).as_str(),
            "h-17636c9ec78323ab9bd295ae7d8a40c1"
        );
        // The same source names a trusted tenant's session as it is, and another untrusted tenant's own one.
        assert_eq!(
            resolve_session(&named("voice", true), &claude, None).as_str(),
            "voice-RM_victim"
        );
        assert_ne!(
            resolve_session(&named("other", false), &claude, None),
            resolve_session(&bench, &claude, None)
        );
    }

    #[test]
    fn classes_are_honored_defaulted_and_clamped() {
        use SloClass::*;
        let agents = tenant(true, Background, Interactive);
        assert_eq!(resolve_class(&agents, None).unwrap(), (Background, false));
        assert_eq!(
            resolve_class(&agents, header("interactive").as_ref()).unwrap(),
            (Interactive, false)
        );
        assert_eq!(
            resolve_class(&agents, header("realtime").as_ref()).unwrap(),
            (Interactive, true)
        );
        let err = resolve_class(&agents, header("urgent").as_ref()).unwrap_err();
        assert!(err.to_string().contains("\"urgent\""));
    }

    #[test]
    fn bodies_lose_engine_fields_and_gain_the_salt() {
        let metrics = Metrics::new();
        let body = br#"{"model":"m","pri\u006frity":-9,"messages":[{"role":"user","content":"hi \"priority\""}],
            "session_id":"x","kv_transfer_params":{"a":1},"ec_transfer_params":{"b":2},"cache_salt":"theirs",
            "request_id":"r",
            "vllm_xargs":{"session_id":"y"},"rid":"theirs","extra_key":"other-tenant",
            "session_params":{"id":"s"},"custom_labels":{"x":"y"},"routed_dp_rank":1,"data_parallel_rank":1,
            "disagg_prefill_dp_rank":0,"bootstrap_host":"h","bootstrap_port":1,"bootstrap_room":7,
            "prompt_cache_key":"pck-1","stream":true}"#;
        let chat = uncapped(Route::ChatCompletions);
        let out = rewrite_body(body, "0123", None, chat, &metrics).unwrap();
        let value: Value = serde_json::from_slice(&out.json).unwrap();
        assert_eq!(
            value,
            json!({"model": "m", "messages": [{"role": "user", "content": "hi \"priority\""}],
                   "prompt_cache_key": "pck-1", "stream": true, "cache_salt": "0123"})
        );
        assert_eq!(out.prompt_cache_key.as_deref(), Some("pck-1"));
        for field in headers::ENGINE_CONTROL_BODY_FIELDS {
            assert_eq!(metrics.sanitized_count(field), 1, "{field}");
        }
        // For an upstream that reads the priority from the body, the gateway's replaces the client's.
        let out = rewrite_body(body, "0123", Some(2), chat, &metrics).unwrap();
        let value: Value = serde_json::from_slice(&out.json).unwrap();
        assert_eq!(value["priority"], 2);
        assert_eq!(metrics.sanitized_count("priority"), 2);
    }

    /// The bounds of a request of `route` whose class has no cap, at one choice.
    fn uncapped(route: Route) -> Bounds {
        Bounds {
            route,
            cap: None,
            max_choices: 1,
        }
    }

    /// The bounds of a request of `route` whose class caps its reply at `cap`, at one choice.
    fn capped(route: Route, cap: u32) -> Bounds {
        Bounds {
            cap: Some(cap),
            ..uncapped(route)
        }
    }

    #[test]
    fn a_capped_body_gets_its_routes_own_limits_at_the_lower_of_the_two() {
        use Route::*;
        let metrics = Metrics::new();
        let rewrite = |body: &str, route, cap| -> Value {
            let out =
                rewrite_body(body.as_bytes(), "s", None, capped(route, cap), &metrics).unwrap();
            serde_json::from_slice(&out.json).unwrap()
        };
        // Chat Completions: max_completion_tokens and max_tokens, both at the lower of the client's limit (the first
        // of them that is not null) and the cap, and none of max_output_tokens, which its engine ignores: a body with
        // only that field once went uncapped.
        for (body, limit) in [
            (r#"{"model":"m"}"#, 24),
            (r#"{"model":"m","max_output_tokens":4000}"#, 24),
            (r#"{"model":"m","max_output_tokens":5}"#, 24),
            (r#"{"model":"m","max_tokens":200}"#, 24),
            (r#"{"model":"m","max_tokens":8}"#, 8),
            (
                r#"{"model":"m","max_completion_tokens":12,"max_tokens":200}"#,
                12,
            ),
            (
                r#"{"model":"m","max_completion_tokens":200,"max_tokens":3}"#,
                24,
            ),
            (
                r#"{"model":"m","max_completion_tokens":null,"max_tokens":9}"#,
                9,
            ),
            (
                r#"{"model":"m","max_completion_tokens":null,"max_tokens":null}"#,
                24,
            ),
            (r#"{"model":"m","max_tokens":0}"#, 0),
        ] {
            assert_eq!(
                rewrite(body, ChatCompletions, 24),
                json!({"max_completion_tokens": limit, "max_tokens": limit, "model": "m", "cache_salt": "s"}),
                "{body}"
            );
        }
        // Completions: max_tokens alone, with vLLM's 16 for a body without one and no limit for a null one.
        for (body, limit) in [
            (r#"{"prompt":"p"}"#, 16),
            (r#"{"prompt":"p","max_tokens":null}"#, 24),
            (r#"{"prompt":"p","max_tokens":200}"#, 24),
            (r#"{"prompt":"p","max_tokens":20}"#, 20),
            (
                r#"{"prompt":"p","max_completion_tokens":2,"max_output_tokens":3}"#,
                16,
            ),
            (
                r#"{"prompt":"p","max_completion_tokens":2,"max_tokens":100}"#,
                24,
            ),
        ] {
            assert_eq!(
                rewrite(body, Completions, 24),
                json!({"max_tokens": limit, "prompt": "p", "cache_salt": "s"}),
                "{body}"
            );
        }
        assert_eq!(
            rewrite(r#"{"prompt":"p"}"#, Completions, 8),
            json!({"max_tokens": 8, "prompt": "p", "cache_salt": "s"})
        );
        // The Responses API: max_output_tokens alone.
        for (body, limit) in [
            (r#"{"input":"p"}"#, 24),
            (r#"{"input":"p","max_output_tokens":null}"#, 24),
            (r#"{"input":"p","max_output_tokens":4000}"#, 24),
            (r#"{"input":"p","max_output_tokens":7}"#, 7),
            (
                r#"{"input":"p","max_tokens":7,"max_completion_tokens":7}"#,
                24,
            ),
        ] {
            assert_eq!(
                rewrite(body, Responses, 24),
                json!({"max_output_tokens": limit, "input": "p", "cache_salt": "s"}),
                "{body}"
            );
        }
        // The limits come first, and every other value is copied as it is.
        let body =
            br#"{"messages":[{"role":"user","content":"\"max_tokens\":1"}],"max_tokens":99,"n":1}"#;
        let out = rewrite_body(body, "s", None, capped(ChatCompletions, 24), &metrics).unwrap();
        assert_eq!(
            &out.json[..],
            br#"{"max_completion_tokens":24,"max_tokens":24,"messages":[{"role":"user","content":"\"max_tokens\":1"}],"n":1,"cache_salt":"s"}"#
        );
    }

    #[test]
    fn an_uncapped_body_keeps_its_limits_as_they_are() {
        let metrics = Metrics::new();
        let body = br#"{"max_output_tokens":4000,"max_tokens":null,"max_completion_tokens":7}"#;
        for route in [Route::ChatCompletions, Route::Completions, Route::Responses] {
            let out = rewrite_body(body, "s", None, uncapped(route), &metrics).unwrap();
            assert_eq!(
                &out.json[..],
                br#"{"max_output_tokens":4000,"max_tokens":null,"max_completion_tokens":7,"cache_salt":"s"}"#
            );
        }
    }

    #[test]
    fn a_degraded_reply_is_capped_in_the_body() {
        let capped = |body: &str, route| -> Value {
            serde_json::from_slice(&cap_reply(body.as_bytes(), route, 24)).unwrap()
        };
        // A body without a limit gets the route's own; a smaller one stays, a larger one is lowered, and the other
        // routes' fields go.
        let chat = capped(r#"{"model":"m","cache_salt":"s"}"#, Route::ChatCompletions);
        assert_eq!(
            chat,
            json!({"max_completion_tokens": 24, "max_tokens": 24, "model": "m", "cache_salt": "s"})
        );
        let both = capped(
            r#"{"max_tokens":100,"max_completion_tokens":12,"cache_salt":"s"}"#,
            Route::ChatCompletions,
        );
        assert_eq!(
            both,
            json!({"max_completion_tokens": 12, "max_tokens": 12, "cache_salt": "s"})
        );
        let null = capped(
            r#"{"max_tokens":null,"cache_salt":"s"}"#,
            Route::Completions,
        );
        assert_eq!(null, json!({"max_tokens": 24, "cache_salt": "s"}));
        let completion = capped(r#"{"prompt":"p","cache_salt":"s"}"#, Route::Completions);
        assert_eq!(
            completion,
            json!({"max_tokens": 16, "prompt": "p", "cache_salt": "s"})
        );
        let response = capped(
            r#"{"input":"p","max_output_tokens":4000,"max_tokens":3,"cache_salt":"s"}"#,
            Route::Responses,
        );
        assert_eq!(
            response,
            json!({"max_output_tokens": 24, "input": "p", "cache_salt": "s"})
        );
        // Every other value is copied as it is.
        let body =
            br#"{"messages":[{"role":"user","content":"\"max_tokens\":1"}],"cache_salt":"s"}"#;
        let out = cap_reply(body, Route::ChatCompletions, 24);
        assert_eq!(
            &out[..],
            br#"{"max_completion_tokens":24,"max_tokens":24,"messages":[{"role":"user","content":"\"max_tokens\":1"}],"cache_salt":"s"}"#
        );
    }

    #[test]
    fn a_cap_in_the_rewrite_is_the_cap_of_the_rewritten_body() {
        // The class's cap is applied in rewrite_body's pass, where cap_reply would parse the rewritten body again:
        // the same bytes, whatever the body, route and priority, and degrade's cap over it keeps the lower of the
        // two, as one cap_reply at the lower does.
        let metrics = Metrics::new();
        assert!(
            bounds::REPLY_LIMITS
                .iter()
                .chain(&bounds::CHOICES)
                .all(|field| !headers::ENGINE_CONTROL_BODY_FIELDS.contains(field))
        );
        let bodies = [
            r#"{"model":"m","messages":[{"role":"user","content":"\"max_tokens\":1"}]}"#,
            r#"{"model":"m","max_tokens":100,"max_completion_tokens":12}"#,
            r#"{"max_tokens":null,"prompt":"p","priority":3}"#,
            r#"{"max_output_tokens":4000,"input":"p","cache_salt":"theirs"}"#,
            r#"{"max_completion_tokens":null,"max_tokens":7}"#,
            r#"{"max_tokens":9000,"session_id":"s","prompt_cache_key":"k","stream":true,"n":1}"#,
            r#"{}"#,
            r#"{"max_completion_tokens":0}"#,
        ];
        for body in bodies {
            for route in [Route::ChatCompletions, Route::Completions, Route::Responses] {
                for priority in [None, Some(2)] {
                    let plain =
                        rewrite_body(body.as_bytes(), "s", priority, uncapped(route), &metrics)
                            .unwrap();
                    for cap in [1, 16, 24, 64, 4096] {
                        let folded = rewrite_body(
                            body.as_bytes(),
                            "s",
                            priority,
                            capped(route, cap),
                            &metrics,
                        )
                        .unwrap();
                        let case = format!("{body} {route:?} {priority:?} {cap}");
                        assert_eq!(folded.json, cap_reply(&plain.json, route, cap), "{case}");
                        assert_eq!(folded.prompt_cache_key, plain.prompt_cache_key, "{case}");
                        for degrade in [1, 16, 64, 10_000] {
                            assert_eq!(
                                cap_reply(&folded.json, route, degrade),
                                cap_reply(&plain.json, route, cap.min(degrade)),
                                "{case} {degrade}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn limits_and_choices_the_engine_might_read_otherwise_are_refused() {
        let metrics = Metrics::new();
        let refused = |body: &str, bounds: Bounds| match rewrite_body(
            body.as_bytes(),
            "s",
            None,
            bounds,
            &metrics,
        ) {
            Err(InvalidBody::Bounds(invalid)) => (invalid.code(), invalid.field()),
            other => panic!("{body}: {other:?}"),
        };
        for route in [Route::ChatCompletions, Route::Completions, Route::Responses] {
            for bounds in [uncapped(route), capped(route, 24)] {
                for (body, code, field) in [
                    (
                        r#"{"max_tokens":8,"max_tokens":9000}"#,
                        "duplicate_field",
                        "max_tokens",
                    ),
                    (
                        r#"{"max_completion_tokens":null,"model":"m","max_completion_tokens":5000}"#,
                        "duplicate_field",
                        "max_completion_tokens",
                    ),
                    (r#"{"n":1,"n":8}"#, "duplicate_field", "n"),
                    (
                        r#"{"max_output_tokens":"64"}"#,
                        "invalid_reply_limit",
                        "max_output_tokens",
                    ),
                    (
                        r#"{"max_tokens":64.0}"#,
                        "invalid_reply_limit",
                        "max_tokens",
                    ),
                    (r#"{"max_tokens":-1}"#, "invalid_reply_limit", "max_tokens"),
                    (r#"{"n":0}"#, "invalid_choices", "n"),
                    (r#"{"n":8}"#, "too_many_choices", "n"),
                    (r#"{"best_of":2}"#, "too_many_choices", "best_of"),
                ] {
                    assert_eq!(refused(body, bounds), (code, field), "{body} {route:?}");
                }
                // A key spelled with an escape is the key it spells, as Python's json reads it.
                let escaped = format!("{{\"max_{}u0074okens\":1,\"max_tokens\":9000}}", '\\');
                assert_eq!(refused(&escaped, bounds), ("duplicate_field", "max_tokens"));
            }
        }
        // A deployment that allows more choices takes them, up to its max_choices.
        let four = Bounds {
            max_choices: 4,
            ..uncapped(Route::ChatCompletions)
        };
        for body in [
            r#"{"n":4}"#,
            r#"{"n":4,"use_beam_search":true}"#,
            r#"{"best_of":3}"#,
        ] {
            assert!(rewrite_body(body.as_bytes(), "s", None, four, &metrics).is_ok());
        }
        assert_eq!(refused(r#"{"n":5}"#, four), ("too_many_choices", "n"));
    }

    #[test]
    fn bodies_must_be_json_objects() {
        let metrics = Metrics::new();
        let chat = uncapped(Route::ChatCompletions);
        for body in [
            &b""[..],
            b"[1]",
            b"\"s\"",
            b"{\"a\":",
            b"{} {}",
            b"{\"a\":1,}",
        ] {
            assert!(
                matches!(
                    rewrite_body(body, "s", None, chat, &metrics),
                    Err(InvalidBody::Json(_))
                ),
                "{body:?}"
            );
        }
        let out = rewrite_body(b" { } ", "s", None, chat, &metrics).unwrap();
        assert_eq!(&out.json[..], br#"{"cache_salt":"s"}"#);
        let out = rewrite_body(b"{}", "s", Some(0), chat, &metrics).unwrap();
        assert_eq!(&out.json[..], br#"{"priority":0,"cache_salt":"s"}"#);
    }

    #[test]
    fn headers_lose_engine_inputs_and_hop_by_hop_fields() {
        let mut headers = HeaderMap::new();
        for (name, value) in [
            ("x-vllm-priority", "-100"),
            ("x-smg-priority", "system"),
            ("x-data-parallel-rank", "1"),
            ("x-session-id", "s-1"),
            ("x-request-id", "theirs"),
            ("x-flatshare-class", "realtime"),
            ("x-flatshare-anything", "1"),
            ("x-claude-code-session-id", "cc-1"),
            ("authorization", "Bearer k"),
            ("connection", "keep-alive, x-hop"),
            ("x-hop", "1"),
            ("content-length", "10"),
            ("host", "gateway"),
            ("x-keep", "1"),
        ] {
            headers.append(
                HeaderName::from_static(name),
                HeaderValue::from_static(value),
            );
        }
        // A trusted tenant's session and class are consumed, not counted as removed.
        let trusted = Metrics::new();
        let inputs = sanitize_headers(&mut headers.clone(), true, &trusted);
        assert_eq!(inputs.session_id, header("s-1"));
        assert_eq!(inputs.class, header("realtime"));
        assert_eq!(trusted.sanitized_count("x-session-id"), 0);
        assert_eq!(trusted.sanitized_count("x-flatshare-class"), 0);
        assert_eq!(trusted.sanitized_count("x-vllm-priority"), 1);

        let untrusted = Metrics::new();
        let inputs = sanitize_headers(&mut headers, false, &untrusted);
        assert_eq!((inputs.session_id, inputs.class), (None, None));
        assert_eq!(inputs.claude_code_session_id, header("cc-1"));
        assert_eq!(headers.keys().collect::<Vec<_>>(), ["x-keep"]);
        for field in [
            "x-vllm-priority",
            "x-smg-priority",
            "x-data-parallel-rank",
            "x-session-id",
            "x-request-id",
            "x-flatshare-class",
            OTHER_FLATSHARE_HEADER,
        ] {
            assert_eq!(untrusted.sanitized_count(field), 1, "{field}");
        }
    }

    #[test]
    fn the_class_priority_goes_in_the_configured_headers() {
        let session = SessionId::new("s-1").unwrap();
        let id = HeaderValue::from_static("r-1");
        let priorities = |class, signals| {
            let mut headers = HeaderMap::new();
            set_engine_headers(&mut headers, &session, class, &id, signals);
            let value = |name| headers.get(name).map(|v| v.to_str().unwrap().to_owned());
            (value("x-vllm-priority"), value("x-smg-priority"))
        };
        let both = PrioritySignals {
            vllm_header: true,
            smg_header: true,
            ..PrioritySignals::default()
        };
        for (class, vllm, smg) in [
            (SloClass::Realtime, "0", "system"),
            (SloClass::Interactive, "1", "interactive"),
            (SloClass::Background, "2", "bulk"),
        ] {
            assert_eq!(
                priorities(class, both),
                (Some(vllm.into()), Some(smg.into()))
            );
            assert_eq!(priorities(class, PrioritySignals::default()), (None, None));
        }
    }

    #[test]
    fn hex_is_lowercase_and_ids_differ() {
        assert_eq!(hex(&[0x00, 0x9f, 0xff]), "009fff");
        assert_ne!(random_hex(), random_hex());
        let (a, b) = (request_id(), request_id());
        assert_eq!(a.len(), 32);
        assert_eq!(
            a.as_bytes()[..16],
            b.as_bytes()[..16],
            "one prefix per process"
        );
        assert_ne!(a, b);
    }
}
