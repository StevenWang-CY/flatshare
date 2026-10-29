//! Lifecycle events: `POST /v1/events` from trusted tenants, deduplicated by `(tenant, source, event_id)` within a
//! bounded window, then handed to the admission point, where the lifecycle policy follows the realtime sessions'
//! turns by them, and to an [`EventSink`], which records them ([`LogSink`]).

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use flatshare_proto::LifecycleEvent;
use serde_json::value::RawValue;

use crate::metrics::Metrics;

/// An event the gateway accepted: not a repeat of one seen within the deduplication window.
#[derive(Clone, Debug)]
pub struct AcceptedEvent {
    /// The tenant that posted it.
    pub tenant: Arc<str>,
    /// The gateway's receipt time, which orders events (emitters' clocks differ).
    pub received: Instant,
    pub event: LifecycleEvent,
}

/// Where accepted events go.
pub trait EventSink: Send + Sync + std::fmt::Debug + 'static {
    /// Called once per accepted event, in receipt order within a request. It runs on the request path, so it
    /// must not block: a consumer that does real work should queue the event.
    fn accept(&self, event: AcceptedEvent);
}

/// The gateway's sink: one structured log line per event (target `flatshare_gateway::events`).
#[derive(Clone, Copy, Debug, Default)]
pub struct LogSink;

impl EventSink for LogSink {
    fn accept(&self, accepted: AcceptedEvent) {
        let event = &accepted.event;
        tracing::info!(
            target: "flatshare_gateway::events",
            tenant = &*accepted.tenant,
            event_id = event.event_id.as_str(),
            session_id = event.session_id.as_str(),
            source = event.source.as_str(),
            event_type = event.kind.label(),
            at_ms = event.at_ms,
            kind = ?event.kind,
            "lifecycle event"
        );
    }
}

/// Deduplication and dispatch of incoming events.
#[derive(Debug)]
pub(crate) struct Events {
    seen: Mutex<Seen>,
    capacity: usize,
    ttl: Duration,
    sink: Arc<dyn EventSink>,
}

/// Keys seen within the window, in arrival order. Every key has the same lifetime, so arrival order is also
/// expiry order and the oldest key is always at the front.
#[derive(Debug, Default)]
struct Seen {
    keys: HashSet<Arc<str>>,
    order: VecDeque<(Instant, Arc<str>)>,
}

impl Events {
    pub fn new(capacity: usize, ttl: Duration, sink: Arc<dyn EventSink>) -> Self {
        Self {
            seen: Mutex::default(),
            capacity,
            ttl,
            sink,
        }
    }

    /// Records a batch, returning how many events were new and how many were repeats. The new ones go to `admit`
    /// (the admission point) in the order they were accepted, and then to the sink.
    pub fn ingest(
        &self,
        tenant: &Arc<str>,
        events: Vec<LifecycleEvent>,
        metrics: &Metrics,
        admit: impl FnOnce(&[LifecycleEvent]),
    ) -> (usize, usize) {
        let total = events.len();
        let (received, fresh) = {
            let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
            // Taken under the lock, so that the receipt times of two batches are in the order they were accepted.
            let received = Instant::now();
            seen.expire(received, self.ttl);
            let fresh: Vec<LifecycleEvent> = events
                .into_iter()
                .filter(|event| {
                    // Per tenant: any trusted tenant may name any source, and must not make another's events
                    // repeats by posting their ids first. Tenant names hold no NUL.
                    let key = format!("{tenant}\0{}\0{}", event.source.as_str(), event.event_id);
                    let fresh = seen.insert(key.into(), received, self.capacity);
                    metrics.event(event.kind.label(), event.source.as_str(), !fresh);
                    fresh
                })
                .collect();
            // Still under the lock, so that a batch accepted after this one reaches the admission point after it.
            // `admit` is synchronous, and the only lock it takes is the admission point's, under which nothing takes
            // this one.
            admit(&fresh);
            (received, fresh)
        };
        let accepted = fresh.len();
        for event in fresh {
            self.sink.accept(AcceptedEvent {
                tenant: Arc::clone(tenant),
                received,
                event,
            });
        }
        (accepted, total - accepted)
    }
}

impl Seen {
    fn expire(&mut self, now: Instant, ttl: Duration) {
        while let Some((at, _)) = self.order.front() {
            if now.duration_since(*at) < ttl {
                break;
            }
            if let Some((_, key)) = self.order.pop_front() {
                self.keys.remove(&key);
            }
        }
    }

    /// Remembers `key`, evicting the oldest key when full. Returns false if it was already present.
    fn insert(&mut self, key: Arc<str>, now: Instant, capacity: usize) -> bool {
        if self.keys.contains(&key) {
            return false;
        }
        if self.order.len() >= capacity
            && let Some((_, oldest)) = self.order.pop_front()
        {
            self.keys.remove(&oldest);
        }
        self.keys.insert(Arc::clone(&key));
        self.order.push_back((now, key));
        true
    }
}

/// The events of a `POST /v1/events` body: one event object or an array of them, each with an `event_id` of 1 to
/// [`LifecycleEvent::MAX_ID_LEN`] bytes.
pub(crate) fn parse(body: &[u8]) -> Result<Vec<LifecycleEvent>, String> {
    let events = if body.trim_ascii_start().starts_with(b"[") {
        let items: Vec<&RawValue> =
            serde_json::from_slice(body).map_err(|e| format!("invalid JSON array: {e}"))?;
        items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                serde_json::from_str(item.get()).map_err(|e| format!("event {i}: {e}"))
            })
            .collect::<Result<Vec<LifecycleEvent>, _>>()?
    } else {
        vec![serde_json::from_slice(body).map_err(|e| format!("invalid event: {e}"))?]
    };
    for (i, event) in events.iter().enumerate() {
        if event.event_id.is_empty() || event.event_id.len() > LifecycleEvent::MAX_ID_LEN {
            return Err(format!(
                "event {i}: event_id must be 1 to {} bytes",
                LifecycleEvent::MAX_ID_LEN
            ));
        }
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_forgets_by_age_and_by_capacity() {
        let mut seen = Seen::default();
        let start = Instant::now();
        let ttl = Duration::from_secs(600);
        assert!(seen.insert("a".into(), start, 2));
        assert!(!seen.insert("a".into(), start, 2));
        assert!(seen.insert("b".into(), start + Duration::from_secs(1), 2));
        // Full: "c" evicts the oldest key, so "a" counts as new again.
        assert!(seen.insert("c".into(), start + Duration::from_secs(2), 2));
        assert!(seen.insert("a".into(), start + Duration::from_secs(3), 2));
        assert_eq!(seen.keys.len(), 2);

        seen.expire(start + Duration::from_secs(602), ttl);
        assert_eq!(
            seen.order.len(),
            1,
            "only the key inserted at +3 s is inside the window"
        );
        assert!(seen.insert("b".into(), start + Duration::from_secs(602), 2));
    }

    /// A batch of one event of the voice session, with id `id`.
    fn batch(id: &str) -> Vec<LifecycleEvent> {
        let json = format!(
            r#"{{"event_id":"{id}","session_id":"voice-1","at_ms":1,"source":"voice-worker","type":"user_turn_ended"}}"#
        );
        parse(json.as_bytes()).unwrap()
    }

    /// Two posts for one session at once: the second is accepted while the first's events are still on their way
    /// to the admission point, which must get them first all the same.
    #[test]
    fn a_sessions_events_reach_the_admission_point_in_the_order_they_were_accepted() {
        let events = Events::new(16, Duration::from_secs(60), Arc::new(LogSink));
        let metrics = Metrics::new();
        let tenant: Arc<str> = "voice".into();
        let reached = Mutex::new(Vec::new());
        let record = |fresh: &[LifecycleEvent]| {
            let ids = fresh.iter().map(|e| e.event_id.clone());
            reached.lock().unwrap().extend(ids);
        };
        let (first_accepted, accepted) = std::sync::mpsc::channel();
        let (second_done, done) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let (events, tenant, metrics, record) = (&events, &tenant, &metrics, &record);
            s.spawn(move || {
                events.ingest(tenant, batch("first"), metrics, |fresh| {
                    first_accepted.send(()).unwrap();
                    // Room for the second post to overtake, which it must not.
                    let _ = done.recv_timeout(Duration::from_millis(200));
                    record(fresh);
                })
            });
            accepted.recv().unwrap();
            s.spawn(move || {
                events.ingest(tenant, batch("second"), metrics, record);
                // The first post may have stopped waiting.
                let _ = second_done.send(());
            });
        });
        assert_eq!(*reached.lock().unwrap(), ["first", "second"]);
    }

    #[test]
    fn errors_name_the_problem() {
        let unknown = parse(br#"{"event_id":"e","session_id":"s","at_ms":1,"source":"client","type":"user_laughed"}"#)
            .unwrap_err();
        assert!(unknown.contains("user_laughed"), "{unknown}");
        let second = parse(
            br#"[{"event_id":"e","session_id":"s","at_ms":1,"source":"client","type":"turn_started"},
                 {"event_id":"f","session_id":"s","at_ms":1,"source":"client"}]"#,
        )
        .unwrap_err();
        assert!(
            second.starts_with("event 1:") && second.contains("type"),
            "{second}"
        );
        let long = format!(
            r#"{{"event_id":"{}","session_id":"s","at_ms":1,"source":"client","type":"turn_started"}}"#,
            "e".repeat(129)
        );
        assert!(
            parse(long.as_bytes())
                .unwrap_err()
                .contains("1 to 128 bytes")
        );
    }
}
