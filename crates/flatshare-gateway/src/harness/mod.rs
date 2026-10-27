//! E2's experiment harness: the admission point and the policies it runs, E2's baselines and the lifecycle policy
//! (which E2's arm 9 runs with make-room), the realtime sessions' turns that policy follows, and their configuration
//! (`[admission]`) and metric families.
//!
//! The configuration to deploy runs none of it (ADR-0006: the measured configuration ran without an admission
//! policy). Without an `[admission]` policy the gateway has no admission point and registers none of these metric
//! families, and under another policy than the lifecycle policy none of that policy's. With one, the proxy reaches
//! the harness only through a request's [`Ticket`], and the lifecycle events through [`Admission::events`]. The
//! caps a deployment may set on each class's replies (`[classes.<class>] max_reply_tokens`, `realtime_max_tokens`)
//! and its bound on choices (`max_choices`) are the proxy's, not the harness's.

mod admission;
pub(crate) mod config;
mod metrics;
mod realtime;
mod sessions;

pub(crate) use admission::{Admission, Ticket};
pub(crate) use metrics::{Metrics, Refusal};
