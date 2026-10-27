//! `flatshare-gateway`: the Flatshare harness in front of one vLLM engine.
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
//! - **a reply cap** on every realtime request (`realtime_max_tokens`), without an admission policy as with one;
//! - **relay**: responses pass through byte for byte, with backpressure, client-disconnect cancellation and a
//!   stall timeout;
//! - **lifecycle events** at `POST /v1/events`, deduplicated, then to the admission point and an [`EventSink`];
//! - **metrics** at `GET /metrics` and one structured log line per request.
//!
//! The admission policies are E2's baselines and the lifecycle policy, which E2's arm 9 runs with make-room.

mod auth;
pub mod config;
mod events;
mod exchange;
pub mod handover;
mod harness;
mod metrics;
mod proxy;
mod relay;
mod sanitize;
mod server;
pub mod systemd;
mod upstream;
mod usage;

pub use config::{Config, ConfigError, ListenAddr};
pub use events::{AcceptedEvent, EventSink, LogSink};
pub use server::{HandOver, MetricsReader, Server};
pub use systemd::Inherited;
