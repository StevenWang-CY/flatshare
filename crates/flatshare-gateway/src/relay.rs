//! The relay of a response: a pump task reads the upstream body into a bounded channel, and the client's response
//! body reads the channel.
//!
//! The channel is the backpressure: when it is full, the pump stops reading the upstream, and if it stays full for
//! the stall timeout the pump aborts both sides. When the client goes away, its body (the receiver) is dropped, the
//! pump sees the channel close at once and drops the upstream body, which closes the upstream connection; vLLM
//! aborts a request whose client disconnects. The pump also ends the response when the admission policy sheds the
//! request, at a drain's cut for a background one, and when one of the request's deadlines passes: its first content
//! by its first-content deadline, each later frame within the idle timeout of the pump's turning back to the upstream,
//! and the whole response by its total ([`crate::deadline`]). A response cut short never ends cleanly: the client's
//! body ends with an error, after the error event of vLLM's own streams on an event stream.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, ready};
use std::time::{Duration, Instant};

use axum::body::Body;
use bytes::Bytes;
use http::HeaderMap;
use http::header::CONTENT_TYPE;
use http_body::Frame;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, watch};

use crate::deadline::{Deadlines, Timeout};
use crate::drain::{self, drain_message};
use crate::exchange::Exchange;
use crate::harness::Ticket;
use crate::state::Shared;
use crate::types::{ErrorEvent, Outcome, UpstreamError};
use crate::upstream::causes;
use crate::usage::Scanner;

/// Starts relaying the upstream body of a response with `headers` to the client, and returns the client's body.
/// `first` is the response's first frame when it was read before the client was answered (preempt-resubmit). An
/// admitted request's `ticket` goes with the relay and is given back when it ends; `cut` is what a request of a
/// class that a drain cuts waits on.
pub(crate) fn start(
    shared: &Shared,
    headers: &HeaderMap,
    upstream: Incoming,
    first: Option<Bytes>,
    ex: Exchange,
    ticket: Option<Ticket>,
    cut: Option<watch::Receiver<bool>>,
) -> Body {
    let mut scanner = Scanner::for_content_type(headers.get(CONTENT_TYPE));
    let (frames, receiver) = mpsc::channel(shared.stream_buffer_frames);
    if let Some(first) = first {
        scanner.feed(&first);
        frames
            .try_send(first)
            .expect("a new channel has room for a frame");
    }
    let aborted = Arc::new(AtomicBool::new(false));
    let pump = Pump {
        upstream,
        frames,
        aborted: Arc::clone(&aborted),
        ex,
        scanner,
        stall_timeout: shared.stall_timeout,
        sheddable: ticket.as_ref().is_some_and(Ticket::sheddable),
        ticket,
        cut,
        grace: shared.drain.grace,
    };
    tokio::spawn(pump.run());
    Body::new(ClientBody {
        frames: receiver,
        aborted,
        unflushed: false,
    })
}

/// The task that reads one upstream body into the client's channel.
struct Pump {
    upstream: Incoming,
    frames: mpsc::Sender<Bytes>,
    /// Set before the pump stops on a stall, an upstream error, a shed or a drain's cut.
    aborted: Arc<AtomicBool>,
    ex: Exchange,
    scanner: Scanner,
    stall_timeout: Duration,
    ticket: Option<Ticket>,
    /// Whether the admission policy may shed the request.
    sheddable: bool,
    /// What the request waits on for a drain's cut, and the grace it had.
    cut: Option<watch::Receiver<bool>>,
    grace: Duration,
}

impl Pump {
    /// Relays until either side ends, until the admission policy sheds the request, or until a drain's cut ends a
    /// background one.
    async fn run(mut self) {
        let _active = self
            .ex
            .resolved
            .class
            .map(|class| self.ex.metrics().stream_started(class));
        let deadlines = self.ex.resolved.deadlines;
        let outcome = loop {
            // The first content by the request's deadline for it, every frame after within the idle timeout.
            let (deadline, timeout) =
                next_deadline(deadlines.as_ref(), self.ex.first_byte.is_some());
            let frame = tokio::select! {
                biased;
                () = self.frames.closed() => break Outcome::Cancelled,
                () = shed(&mut self.ticket), if self.sheddable => break Outcome::Shed,
                () = drain::cut(&mut self.cut) => {
                    self.end_with(ErrorEvent::Draining, drain_message(self.grace)).await;
                    break Outcome::Drained;
                }
                () = sleep_until(deadline) => {
                    self.ex.metrics().upstream_error(UpstreamError::Timeout);
                    let message = deadlines.map(|d| d.message(timeout)).unwrap_or_default();
                    self.end_with(ErrorEvent::Timeout, message).await;
                    break Outcome::timeout(timeout);
                }
                frame = self.upstream.frame() => frame,
            };
            let data = match frame {
                None => break Outcome::Completed,
                Some(Err(err)) => {
                    self.ex.metrics().upstream_error(UpstreamError::Body);
                    // With [[nodes]], a node whose stream broke off is probed at once: its engine may be gone.
                    self.ex.broke();
                    let message = format!("the upstream stream broke off: {}", causes(&err));
                    self.end_with(ErrorEvent::Upstream, message).await;
                    break Outcome::TransportError;
                }
                // Trailers are not relayed; vLLM sends none.
                Some(Ok(frame)) => match frame.into_data() {
                    Ok(data) => data,
                    Err(_) => continue,
                },
            };
            if self.ex.first_byte.is_none() {
                self.ex.first_byte = Some(self.ex.received.elapsed());
                // A 2xx response's first content: the engine generates.
                if self.ex.status.is_some_and(|status| status.is_success()) {
                    self.ex.succeeded();
                }
                // A realtime request's first byte frees the background work preempt-resubmit holds back for it.
                if let Some(ticket) = &mut self.ticket {
                    ticket.first_byte();
                }
            }
            let scanned = data.clone();
            match self.frames.try_send(data) {
                Ok(()) => {}
                Err(TrySendError::Closed(_)) => break Outcome::Cancelled,
                Err(TrySendError::Full(data)) => {
                    if let Err(outcome) = self.send_full(data).await {
                        break outcome;
                    }
                }
            }
            self.scanner.feed(&scanned);
        };
        let Self {
            upstream,
            frames,
            aborted,
            mut ex,
            scanner,
            ticket,
            ..
        } = self;
        if outcome != Outcome::Completed && outcome != Outcome::Cancelled {
            aborted.store(true, Ordering::Release);
        }
        // An unfinished upstream body closes its connection when dropped, which aborts the engine request.
        drop(upstream);
        drop(frames);
        ex.summary = scanner.finish();
        // The request leaves the admission point once its upstream connection is gone.
        if let Some(mut ticket) = ticket {
            ticket.prompt_tokens = ex.summary.usage.map(|usage| usage.prompt);
            drop(ticket);
        }
        ex.finish(outcome);
    }

    /// Ends an event stream with the error event of `kind`, sent and counted only if the client takes it: a drain's
    /// at once or not at all (a drain does not wait for a client that stops reading), the others within the stall
    /// timeout. `message` goes to the request's log line whatever the response.
    async fn end_with(&mut self, kind: ErrorEvent, message: String) {
        if self.scanner.is_events() {
            let event = error_event(kind, &message, self.scanner.at_event_boundary());
            let sent = match kind {
                ErrorEvent::Draining => self.frames.try_send(event).is_ok(),
                ErrorEvent::Upstream | ErrorEvent::Timeout => {
                    let sent =
                        tokio::time::timeout(self.stall_timeout, self.frames.send(event)).await;
                    matches!(sent, Ok(Ok(())))
                }
            };
            if let (true, Some(class)) = (sent, self.ex.resolved.class) {
                self.ex.metrics().error_event(class, kind);
            }
        }
        self.ex.note_error(message);
    }

    /// Waits for room in the client's full channel to send `data`, for at most the stall timeout. `Err` is how the
    /// response ends instead: a shed, a drain's cut, the request's total, a client that left or one too slow.
    async fn send_full(&mut self, data: Bytes) -> Result<(), Outcome> {
        let deadlines = self.ex.resolved.deadlines;
        let total = deadlines.map(|d| d.total().0);
        let sent = tokio::select! {
            biased;
            () = shed(&mut self.ticket), if self.sheddable => return Err(Outcome::Shed),
            () = drain::cut(&mut self.cut) => {
                self.ex.note_error(drain_message(self.grace));
                return Err(Outcome::Drained);
            }
            // A client too slow for the request's total: the client's channel is full, so no event.
            () = sleep_until(total) => {
                self.ex.note_error(deadlines.map(|d| d.message(Timeout::Total)).unwrap_or_default());
                return Err(Outcome::TimeoutTotal);
            }
            sent = tokio::time::timeout(self.stall_timeout, self.frames.send(data)) => sent,
        };
        match sent {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err(Outcome::Cancelled),
            Err(_) => Err(Outcome::Stalled),
        }
    }
}

/// The error event of vLLM's own streams (`data: {"error": {...}}`), which the gateway sends before it aborts an
/// event stream, so that a client learns why: the upstream broke it off, or a draining gateway cut it. A client
/// that ignores it still sees the broken stream. After a partial event, a blank line ends that event first.
pub(crate) fn error_event(kind: ErrorEvent, message: &str, at_boundary: bool) -> Bytes {
    let error = serde_json::json!({
        "error": {"message": message, "type": kind.as_str(), "param": null, "code": kind.code()},
    });
    let start = if at_boundary { "" } else { "\n\n" };
    Bytes::from(format!("{start}data: {error}\n\n"))
}

/// The deadline of the next upstream frame, from now, and which it is: the request's first-content or total deadline
/// for its first frame, the idle timeout or the total for the others; none for a request without deadlines.
fn next_deadline(deadlines: Option<&Deadlines>, begun: bool) -> (Option<Instant>, Timeout) {
    match deadlines {
        Some(deadlines) if begun => {
            let (at, timeout) = deadlines.next_frame(Instant::now());
            (Some(at), timeout)
        }
        Some(deadlines) => {
            let (at, timeout) = deadlines.first_frame();
            (Some(at), timeout)
        }
        None => (None, Timeout::Total),
    }
}

/// Completes at `deadline`; never without one.
async fn sleep_until(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
        None => std::future::pending().await,
    }
}

/// Completes when the admission policy sheds the request.
async fn shed(ticket: &mut Option<Ticket>) {
    match ticket {
        Some(ticket) => ticket.shed().await,
        None => std::future::pending().await,
    }
}

/// The client's half of a relayed response.
#[derive(Debug)]
struct ClientBody {
    frames: mpsc::Receiver<Bytes>,
    /// Set by the pump before it stops on a stall, an upstream error or a shed, so the client sees an error
    /// instead of a clean end: a truncated response must not look complete.
    aborted: Arc<AtomicBool>,
    /// Whether a frame was handed to hyper since it last found the body pending. hyper drops what it has not
    /// written when a body errors, so an abort waits one poll, in which hyper writes out the last frames, such as
    /// the error event.
    unflushed: bool,
}

#[derive(Debug)]
struct StreamAborted;

impl std::fmt::Display for StreamAborted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the upstream response was aborted")
    }
}

impl std::error::Error for StreamAborted {}

impl http_body::Body for ClientBody {
    type Data = Bytes;
    type Error = StreamAborted;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, StreamAborted>>> {
        let polled = self.frames.poll_recv(cx);
        if polled.is_pending() {
            self.unflushed = false;
        }
        match ready!(polled) {
            Some(data) => {
                self.unflushed = true;
                Poll::Ready(Some(Ok(Frame::data(data))))
            }
            // The pump stores the flag before it drops the sender, so it is visible once the channel closes.
            None if self.aborted.load(Ordering::Acquire) => {
                if std::mem::take(&mut self.unflushed) {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Poll::Ready(Some(Err(StreamAborted)))
            }
            None => Poll::Ready(None),
        }
    }
}
