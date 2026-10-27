//! The relay of a response: a pump task reads the upstream body into a bounded channel, and the client's response
//! body reads the channel.
//!
//! The channel is the backpressure: when it is full, the pump stops reading the upstream, and if it stays full for
//! the stall timeout the pump aborts both sides. When the client goes away, its body (the receiver) is dropped, the
//! pump sees the channel close at once and drops the upstream body, which closes the upstream connection; vLLM
//! aborts a request whose client disconnects. The pump also ends the response when the admission policy sheds the
//! request, or at a drain's cut for a background one. A response cut short never ends cleanly: the client's body
//! ends with an error, after the error event of vLLM's own streams on an event stream.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, ready};
use std::time::Duration;

use axum::body::Body;
use bytes::Bytes;
use http::HeaderMap;
use http::header::CONTENT_TYPE;
use http_body::Frame;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, watch};

use crate::exchange::Exchange;
use crate::harness::Ticket;
use crate::metrics::{ErrorEvent, Outcome, UpstreamError};
use crate::server::{self, Shared};
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
    /// What the request waits on for a drain's cut, and the grace it had.
    cut: Option<watch::Receiver<bool>>,
    grace: Duration,
}

impl Pump {
    /// Relays until either side ends, until the admission policy sheds the request, or until a drain's cut ends a
    /// background one.
    async fn run(self) {
        let Self {
            mut upstream,
            frames,
            aborted,
            mut ex,
            mut scanner,
            stall_timeout,
            mut ticket,
            mut cut,
            grace,
        } = self;
        let _active = ex.class.map(|class| ex.metrics().stream_started(class));
        let sheddable = ticket.as_ref().is_some_and(Ticket::sheddable);
        let outcome = loop {
            let frame = tokio::select! {
                biased;
                () = frames.closed() => break Outcome::Cancelled,
                () = shed(&mut ticket), if sheddable => break Outcome::Shed,
                () = server::cut(&mut cut) => {
                    let message = server::drain_message(grace);
                    // At once or not at all: a drain does not wait for a client that stops reading.
                    if scanner.is_events()
                        && frames
                            .try_send(error_event(ErrorEvent::Draining, &message, scanner.at_event_boundary()))
                            .is_ok()
                        && let Some(class) = ex.class
                    {
                        ex.metrics().error_event(class, ErrorEvent::Draining);
                    }
                    ex.note_error(message);
                    break Outcome::Drained;
                }
                frame = upstream.frame() => frame,
            };
            let data = match frame {
                None => break Outcome::Completed,
                Some(Err(err)) => {
                    ex.metrics().upstream_error(UpstreamError::Body);
                    let message = format!("the upstream stream broke off: {}", causes(&err));
                    if scanner.is_events() {
                        // Within the stall timeout: a client that stops reading gets none.
                        let event = error_event(
                            ErrorEvent::Upstream,
                            &message,
                            scanner.at_event_boundary(),
                        );
                        let sent = tokio::time::timeout(stall_timeout, frames.send(event)).await;
                        if let (Ok(Ok(())), Some(class)) = (sent, ex.class) {
                            ex.metrics().error_event(class, ErrorEvent::Upstream);
                        }
                    }
                    ex.note_error(message);
                    break Outcome::Error;
                }
                // Trailers are not relayed; vLLM sends none.
                Some(Ok(frame)) => match frame.into_data() {
                    Ok(data) => data,
                    Err(_) => continue,
                },
            };
            if ex.first_byte.is_none() {
                ex.first_byte = Some(ex.received.elapsed());
                // A realtime request's first byte frees the background work preempt-resubmit holds back for it.
                if let Some(ticket) = &mut ticket {
                    ticket.first_byte();
                }
            }
            let scanned = data.clone();
            if let Err(err) = frames.try_send(data) {
                let data = match err {
                    TrySendError::Full(data) => data,
                    TrySendError::Closed(_) => break Outcome::Cancelled,
                };
                let sent = tokio::select! {
                    biased;
                    () = shed(&mut ticket), if sheddable => break Outcome::Shed,
                    () = server::cut(&mut cut) => {
                        ex.note_error(server::drain_message(grace));
                        break Outcome::Drained;
                    }
                    sent = tokio::time::timeout(stall_timeout, frames.send(data)) => sent,
                };
                match sent {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) => break Outcome::Cancelled,
                    Err(_) => break Outcome::Stalled,
                }
            }
            scanner.feed(&scanned);
        };
        if matches!(
            outcome,
            Outcome::Stalled | Outcome::Error | Outcome::Shed | Outcome::Drained
        ) {
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
}

/// The error event of vLLM's own streams (`data: {"error": {...}}`), which the gateway sends before it aborts an
/// event stream, so that a client learns why: the upstream broke it off, or a draining gateway cut it. A client
/// that ignores it still sees the broken stream. After a partial event, a blank line ends that event first.
fn error_event(kind: ErrorEvent, message: &str, at_boundary: bool) -> Bytes {
    let error = serde_json::json!({
        "error": {"message": message, "type": kind.as_str(), "param": null, "code": kind.code()},
    });
    let start = if at_boundary { "" } else { "\n\n" };
    Bytes::from(format!("{start}data: {error}\n\n"))
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
