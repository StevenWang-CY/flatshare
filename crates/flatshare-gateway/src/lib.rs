//! `flatshare-gateway`: Flatshare's serving control plane in front of one vLLM engine, or of several GPU nodes, with
//! E2's admission policies as its harness ([`harness`]), which only an `[admission]` table runs.
//!
//! It carries Flatshare's request contract and nothing a stock proxy would not need:
//! - **auth**: tenants from API-key digests (constant-time match), or from the listener for a private Unix
//!   socket;
//! - **sanitization**: the headers and body fields the engine, or a router in front of it, trusts for
//!   scheduling and accounting (`flatshare_proto::headers`) are removed from client requests, and the gateway sets its
//!   own;
//! - **session and class**: resolved per PROTOCOL.md and sent as `X-Session-ID` and a priority where the
//!   upstream reads one (`X-Vllm-Priority`, the body's `priority` or `X-SMG-Priority`), with a per-tenant
//!   `cache_salt`;
//! - **admission**, E2's harness (`harness`): a `flatshare_core::admission` policy decides when a request goes
//!   upstream, and the gateway does the waiting and the cancellation; under the lifecycle policy it also follows
//!   the realtime sessions' turns from lifecycle events and realtime requests;
//! - **bounds on what a request asks for**: a reply cap per class (`[classes.<class>] max_reply_tokens`), written in
//!   each route's own fields as vLLM reads them, and at most `max_choices` choices, without an admission policy as
//!   with one;
//! - **intake bounds**, without an admission policy as with one: requests in flight in all, by class and by tenant,
//!   taken before a body is read and refused at once, and a budget for the bodies held; a reload's successor counts
//!   its predecessor's ([`ledger`]);
//! - **relay**: responses pass through byte for byte, with backpressure, client-disconnect cancellation and a
//!   stall timeout;
//! - **nodes** (`[[nodes]]`, [`nodes`]): several GPU nodes, each judged as a whole pipeline (its engine's canary and
//!   readiness, its STT and TTS servers), sessions placed on them stickily with call slots and agents' bounds, and
//!   failover: a session whose node fails moves by its next request, a realtime request whose node fails before its
//!   first content is sent once more elsewhere, and a node that carries calls moved to it defers its agents' requests
//!   until those calls end;
//! - **lifecycle events** at `POST /v1/events`, deduplicated, then to the admission point and an [`EventSink`];
//! - **metrics** at `GET /metrics` and one structured log line per request.
//!
//! The admission policies are E2's baselines and the lifecycle policy, which E2's arm 9 runs with make-room.

mod auth;
mod bounds;
mod canary;
pub mod config;
mod deadline;
mod engine;
mod events;
mod exchange;
pub mod handover;
mod harness;
mod intake;
mod ledger;
mod metrics;
mod nodes;
mod proxy;
mod relay;
mod sanitize;
mod server;
pub mod systemd;
mod types;
mod upstream;
mod usage;
mod util;

pub use config::{Config, ConfigError, ListenAddr};
pub use events::{AcceptedEvent, EventSink, LogSink};
pub use ledger::{Followed, Ledger};
pub use server::{HandOver, MetricsReader, Server};
pub use systemd::Inherited;
