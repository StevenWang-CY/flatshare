//! A request's deadlines, from its class's `[classes.<class>]` table: one for each phase it waits through, and two
//! logical ones, its first content and its total, counted from its receipt, which hold across the attempts of a
//! request that preempt-resubmit sends again, so that a retry never buys a request more time. Every wait of a request
//! is bounded by the earliest deadline that applies to it, and the one that passes first is the request's outcome.
//!
//! What a client gets depends on the phase it was in: 408 for an upload that did not finish, 503 with `Retry-After`
//! for a wait at the admission point (the engine never saw the request), 504 for an upstream that did not answer, and,
//! once a response has begun, the error event of vLLM's streams and an abort.

use std::time::{Duration, Instant};

use flatshare_proto::SloClass;

use crate::config::ClassConfig;

/// Which deadline passed: the outcome's kind, and the configuration key that set it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Timeout {
    /// The body, from the headers' receipt (`upload_timeout_s`).
    Upload,
    /// A wait at the admission point (`queue_timeout_s`).
    Queue,
    /// An attempt's wait for the response headers (`headers_timeout_s`).
    Headers,
    /// The first body frame, from the receipt (`first_content_timeout_s`).
    FirstContent,
    /// The wait between two body frames (`idle_timeout_s`).
    Idle,
    /// The whole request, from its receipt to its response's end (`total_timeout_s`).
    Total,
}

impl Timeout {
    /// The configuration key of the deadline.
    pub const fn key(self) -> &'static str {
        match self {
            Self::Upload => "upload_timeout_s",
            Self::Queue => "queue_timeout_s",
            Self::Headers => "headers_timeout_s",
            Self::FirstContent => "first_content_timeout_s",
            Self::Idle => "idle_timeout_s",
            Self::Total => "total_timeout_s",
        }
    }
}

/// The phase a request was in when a deadline passed, which decides its answer when no response has begun.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Reading its body: 408.
    Upload,
    /// Waiting at the admission point, before any attempt or between two: 503 with `Retry-After`.
    Queue,
    /// Waiting for the upstream's response: 504.
    Upstream,
}

/// A request's deadlines.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Deadlines {
    class: SloClass,
    received: Instant,
    upload: Duration,
    queue: Duration,
    headers: Duration,
    first_content: Duration,
    idle: Duration,
    total: Duration,
}

impl Deadlines {
    /// The deadlines of a request of `class` received at `received`.
    pub fn new(class: SloClass, config: &ClassConfig, received: Instant) -> Self {
        Self {
            class,
            received,
            upload: config.upload_timeout,
            queue: config.queue_timeout,
            headers: config.headers_timeout,
            first_content: config.first_content_timeout,
            idle: config.idle_timeout,
            total: config.total_timeout,
        }
    }

    /// The deadline of the body's last byte.
    pub fn upload(&self) -> (Instant, Timeout) {
        earliest([(self.received + self.upload, Timeout::Upload), self.total()])
    }

    /// The deadline of a wait at the admission point that begins `now`.
    pub fn queue(&self, now: Instant) -> (Instant, Timeout) {
        earliest([
            (now + self.queue, Timeout::Queue),
            self.first_content(),
            self.total(),
        ])
    }

    /// The deadline of an attempt's response headers, for an attempt sent `now`.
    pub fn headers(&self, now: Instant) -> (Instant, Timeout) {
        earliest([
            (now + self.headers, Timeout::Headers),
            self.first_content(),
            self.total(),
        ])
    }

    /// The deadline of the first body frame, whatever the attempt.
    pub fn first_frame(&self) -> (Instant, Timeout) {
        earliest([self.first_content(), self.total()])
    }

    /// The deadline of the body frame after one that arrived at `last`.
    pub fn next_frame(&self, last: Instant) -> (Instant, Timeout) {
        earliest([(last + self.idle, Timeout::Idle), self.total()])
    }

    /// The deadline of the whole request.
    pub fn total(&self) -> (Instant, Timeout) {
        (self.received + self.total, Timeout::Total)
    }

    fn first_content(&self) -> (Instant, Timeout) {
        (self.received + self.first_content, Timeout::FirstContent)
    }

    /// Why `timeout` ended a request, for its client and its log line.
    pub fn message(&self, timeout: Timeout) -> String {
        let class = self.class;
        let (what, limit) = match timeout {
            Timeout::Upload => ("the request body did not arrive", self.upload),
            Timeout::Queue => ("the request was not admitted", self.queue),
            Timeout::Headers => ("the upstream sent no response headers", self.headers),
            Timeout::FirstContent => ("the upstream sent no first content", self.first_content),
            Timeout::Idle => ("the upstream sent nothing more", self.idle),
            Timeout::Total => ("the request did not end", self.total),
        };
        format!(
            "{what} within classes.{class}.{} ({} s)",
            timeout.key(),
            limit.as_secs_f64()
        )
    }
}

/// The earliest of some deadlines, the first of equal ones.
fn earliest<const N: usize>(deadlines: [(Instant, Timeout); N]) -> (Instant, Timeout) {
    deadlines
        .into_iter()
        .reduce(|a, b| if b.0 < a.0 { b } else { a })
        .expect("at least one deadline")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ClassConfig {
        ClassConfig {
            upload_timeout: Duration::from_secs(5),
            queue_timeout: Duration::from_secs(10),
            headers_timeout: Duration::from_secs(20),
            first_content_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(4),
            total_timeout: Duration::from_secs(60),
            ..ClassConfig::defaults(SloClass::Background)
        }
    }

    #[test]
    fn each_wait_is_bounded_by_the_earliest_deadline_that_applies() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let d = Deadlines::new(SloClass::Background, &config(), t0);
        assert_eq!(d.upload(), (at(5), Timeout::Upload));
        assert_eq!(d.queue(at(1)), (at(11), Timeout::Queue));
        // A request sent at 1 s waits for its headers to 21 s; one sent again at 15 s, to its first content's 30 s.
        assert_eq!(d.headers(at(1)), (at(21), Timeout::Headers));
        assert_eq!(d.headers(at(15)), (at(30), Timeout::FirstContent));
        // A wait at the admission point that begins late is cut by the request's own deadlines.
        assert_eq!(d.queue(at(25)), (at(30), Timeout::FirstContent));
        assert_eq!(d.first_frame(), (at(30), Timeout::FirstContent));
        assert_eq!(d.next_frame(at(40)), (at(44), Timeout::Idle));
        assert_eq!(d.next_frame(at(58)), (at(60), Timeout::Total));
        // A total shorter than a phase's bound cuts that phase.
        let short = ClassConfig {
            total_timeout: Duration::from_secs(3),
            ..config()
        };
        let d = Deadlines::new(SloClass::Realtime, &short, t0);
        assert_eq!(d.upload(), (at(3), Timeout::Total));
        assert_eq!(d.first_frame(), (at(3), Timeout::Total));
        assert_eq!(
            d.message(Timeout::Total),
            "the request did not end within classes.realtime.total_timeout_s (3 s)"
        );
    }
}
