//! What a gateway holds on its engine, for a reload's successor to count with its own: during a reload the
//! predecessor and the successor serve the same engine, and each would otherwise bound and admit as if it were alone.
//!
//! One admission point owns the admissions from the handover on. The predecessor closes as it hands its sockets over:
//! its intake and its admission point admit nothing more, and its waiting requests get 503 `draining`, so that their
//! clients send them again to the successor ([`crate::intake`], [`crate::harness`]). What it still holds only shrinks
//! as its requests end, and it tells the successor each time: its requests in flight in all, by class and by tenant,
//! the bytes of their bodies, and its admission point's admitted requests, background tokens, realtime requests
//! without a first byte and armed sessions. The successor counts them with its own until the predecessor exits.
//!
//! The channel is a Unix socket in a directory of the predecessor's own, mode 700, in `$XDG_RUNTIME_DIR` (or the
//! temporary directory), whose path the successor finds in its environment (`FLATSHARE_GATEWAY_LEDGER`, set by
//! [`crate::handover`]). The predecessor sends a line of JSON, a [`Load`], whenever what it holds changes, the last one
//! with `closed` set once it has closed, and the connection closes when it exits, which the successor reads as nothing
//! held. The successor waits for `closed`, for a second or two at most, before it serves, so that no request reaches
//! both admission points. With `[[nodes]]` the closing line also hands over the nodes' verdicts and every session's
//! placement ([`crate::nodes::Handed`]), which the successor takes before it serves. A successor that does not connect within the handover's timeout, such as a build from before
//! the ledger, finds nothing: the directory is removed as soon as one connects or the time is up.

use std::io;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::Duration;

use flatshare_core::admission::{Elsewhere, Turns};
use serde::{Deserialize, Serialize};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::watch;

use crate::intake;
use crate::state::Shared;

/// What a gateway holds on its engine, one line of the ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Load {
    /// Its requests past the intake.
    pub intake: intake::Held,
    /// What its admission point admitted; none without an admission policy.
    pub admission: Option<Admitted>,
    /// Whether it admits nothing more, having handed its sockets over.
    pub closed: bool,
    /// With `[[nodes]]`, in the line that closes: its nodes' verdicts and its placements, which the successor takes
    /// over, so that a reload moves no session and finds its nodes as they were. Absent otherwise, so that a gateway
    /// with one upstream writes the lines it always did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nodes: Option<crate::nodes::Handed>,
}

/// What an admission point admitted ([`Elsewhere`]), as the ledger spells it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Admitted {
    pub admitted: [u32; 3],
    pub background_tokens: u64,
    pub realtime_pending: u32,
    pub speaking: u32,
    pub critical: u32,
}

impl From<Elsewhere> for Admitted {
    fn from(held: Elsewhere) -> Self {
        Self {
            admitted: held.admitted,
            background_tokens: held.background_tokens,
            realtime_pending: held.realtime_pending,
            speaking: held.turns.speaking,
            critical: held.turns.critical,
        }
    }
}

impl Load {
    /// What a successor's admission point counts of this load. A predecessor without an admission policy admitted
    /// every request it took in, so its requests in flight count as admitted, by class.
    pub fn elsewhere(&self) -> Elsewhere {
        match self.admission {
            Some(admitted) => Elsewhere {
                admitted: admitted.admitted,
                background_tokens: admitted.background_tokens,
                realtime_pending: admitted.realtime_pending,
                turns: Turns {
                    speaking: admitted.speaking,
                    critical: admitted.critical,
                },
            },
            None => Elsewhere {
                admitted: self
                    .intake
                    .classes
                    .map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
                ..Elsewhere::default()
            },
        }
    }
}

/// A gateway's ledger, for its successors ([`crate::Server::ledger`]).
#[derive(Clone, Debug)]
pub struct Ledger(pub(crate) Weak<Shared>);

impl Ledger {
    /// Opens a channel for one successor and serves it, for as long as `wait` for the successor to connect, and then
    /// until either exits. Returns the path to give the successor.
    pub fn open(&self, wait: Duration) -> io::Result<PathBuf> {
        let channel = Channel::bind()?;
        let path = channel.path.clone();
        tokio::spawn(serve(channel, self.0.clone(), wait));
        Ok(path)
    }
}

/// The predecessor's end: a listening socket in a fresh directory of its own, both removed when it is dropped.
#[derive(Debug)]
struct Channel {
    dir: PathBuf,
    path: PathBuf,
    listener: UnixListener,
}

impl Channel {
    fn bind() -> io::Result<Self> {
        let base = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .filter(|dir| dir.is_absolute())
            .unwrap_or_else(std::env::temp_dir);
        // A fresh name, created with mode 700 (and not if it exists), so that no other user can connect.
        let dir = base.join(format!(
            "flatshare-gateway-ledger-{}",
            &crate::util::random_hex()[..16]
        ));
        std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
        let path = dir.join("ledger.sock");
        match UnixListener::bind(&path) {
            Ok(listener) => Ok(Self {
                dir,
                path,
                listener,
            }),
            Err(err) => {
                let _ = std::fs::remove_dir(&dir);
                Err(err)
            }
        }
    }
}

impl Drop for Channel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir(&self.dir);
    }
}

/// Sends the successor that connects within `wait` a [`Load`] whenever what the gateway holds changes, until the
/// successor goes or the gateway exits.
async fn serve(channel: Channel, shared: Weak<Shared>, wait: Duration) {
    let accepted = tokio::time::timeout(wait, channel.listener.accept()).await;
    drop(channel);
    let Ok(Ok((stream, _))) = accepted else {
        return;
    };
    let Some(changed) = shared.upgrade().map(|shared| Arc::clone(&shared.changed)) else {
        return;
    };
    let mut sent: Option<Load> = None;
    let mut buf = [0; 64];
    loop {
        // Enabled before the load is read, so that a change between the two wakes the next wait.
        let notified = changed.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let Some(load) = shared.upgrade().map(|shared| shared.load()) else {
            return;
        };
        if sent.as_ref() != Some(&load) {
            let mut line = serde_json::to_vec(&load).expect("a load serializes");
            line.push(b'\n');
            if write_all(&stream, &line).await.is_err() {
                return;
            }
            sent = Some(load);
        }
        tokio::select! {
            () = &mut notified => {}
            // The successor sends nothing: a read that ends is the successor gone.
            read = read_some(&stream, &mut buf) => if !matches!(read, Ok(n) if n > 0) {
                return;
            },
        }
    }
}

/// Writes all of `bytes`.
async fn write_all(stream: &UnixStream, mut bytes: &[u8]) -> io::Result<()> {
    while !bytes.is_empty() {
        stream.writable().await?;
        match stream.try_write(bytes) {
            Ok(n) => bytes = &bytes[n..],
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

/// Reads what has arrived into `buf`, once some has: 0 at the end of the stream.
async fn read_some(stream: &UnixStream, buf: &mut [u8]) -> io::Result<usize> {
    loop {
        stream.readable().await?;
        match stream.try_read(buf) {
            Ok(n) => return Ok(n),
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {}
            Err(err) => return Err(err),
        }
    }
}

/// The successor's end ([`crate::Server::follow`]): it completes [`Followed::closed`] once the predecessor has closed
/// or exited.
#[derive(Debug)]
pub struct Followed(watch::Receiver<bool>);

impl Followed {
    /// Completes once the predecessor admits nothing more, or is gone.
    pub async fn closed(&mut self) {
        let _ = self.0.wait_for(|closed| *closed).await;
    }
}

/// Connects to a predecessor's ledger at `path` and counts what it holds with `shared`'s own until it exits.
pub(crate) async fn follow(path: &Path, shared: Weak<Shared>) -> io::Result<Followed> {
    let stream = UnixStream::connect(path).await?;
    let (closed, followed) = watch::channel(false);
    tokio::spawn(read(stream, shared, closed));
    Ok(Followed(followed))
}

/// A ledger's line is a few hundred bytes, and with `[[nodes]]` its closing one holds every placement, at most
/// 4,096 of a few hundred bytes each; a longer one is not a ledger's.
const MAX_LINE: usize = 4 << 20;

async fn read(stream: UnixStream, shared: Weak<Shared>, closed: watch::Sender<bool>) {
    let mut pending = Vec::new();
    let mut buf = [0; 4096];
    'stream: loop {
        let n = match read_some(&stream, &mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        pending.extend_from_slice(&buf[..n]);
        while let Some(end) = pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = pending.drain(..=end).collect();
            let Some(shared) = shared.upgrade() else {
                return;
            };
            match serde_json::from_slice::<Load>(&line) {
                Ok(load) => {
                    shared.set_predecessor(&load);
                    if load.closed {
                        closed.send_replace(true);
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "the predecessor's ledger sent a line that is not a load: {err}"
                    );
                    break 'stream;
                }
            }
        }
        if pending.len() > MAX_LINE {
            tracing::warn!("the predecessor's ledger sent a line longer than {MAX_LINE} bytes");
            break;
        }
    }
    if let Some(shared) = shared.upgrade() {
        shared.set_predecessor(&Load::default());
    }
    closed.send_replace(true);
    tracing::info!("the predecessor is gone: its requests no longer count with this gateway's");
}

/// The tests: a ledger line that reads back, a predecessor without a policy's requests counted as admitted, a closing
/// line that hands placements over while a lone upstream's lines name no nodes.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_load_is_one_line_of_json_that_reads_back_and_takes_what_it_lacks_as_none() {
        let load = Load {
            intake: intake::Held {
                in_flight: 3,
                classes: [1, 0, 2],
                tenants: [("agents".to_owned(), 2), ("voice".to_owned(), 1)].into(),
                body_bytes: 4096,
            },
            admission: Some(Admitted {
                admitted: [1, 0, 2],
                background_tokens: 9000,
                realtime_pending: 1,
                speaking: 1,
                critical: 0,
            }),
            closed: true,
            nodes: None,
        };
        let line = serde_json::to_string(&load).unwrap();
        assert!(!line.contains('\n'));
        assert_eq!(serde_json::from_str::<Load>(&line).unwrap(), load);
        let elsewhere = load.elsewhere();
        assert_eq!(
            (elsewhere.admitted, elsewhere.turns.speaking),
            ([1, 0, 2], 1)
        );
        // A load from a predecessor without a policy counts its requests in flight as admitted.
        let plain = serde_json::from_str::<Load>(r#"{"intake":{"in_flight":2,"classes":[0,1,1]}}"#)
            .unwrap();
        assert_eq!(plain.elsewhere().admitted, [0, 1, 1]);
        assert!(!plain.closed);
    }
    #[test]
    fn a_closing_line_hands_the_placements_over_and_a_lone_upstreams_lines_name_no_nodes() {
        use crate::nodes::{Handed, HandedNode, HandedSession, Verdict};
        let lone = Load {
            closed: true,
            ..Load::default()
        };
        let line = serde_json::to_string(&lone).unwrap();
        assert!(!line.contains("nodes"), "{line}");
        let handed = Load {
            closed: true,
            nodes: Some(Handed {
                nodes: vec![HandedNode {
                    name: "gray".into(),
                    verdict: Verdict::Unhealthy,
                }],
                sessions: vec![HandedSession {
                    tenant: "voice".into(),
                    session: flatshare_proto::SessionId::new("call-1").unwrap(),
                    node: "phastform".into(),
                    class: flatshare_proto::SloClass::Realtime,
                    trusted: true,
                    moved_from: Some("gray".into()),
                }],
            }),
            ..Load::default()
        };
        let line = serde_json::to_string(&handed).unwrap();
        assert!(!line.contains('\n'));
        assert_eq!(serde_json::from_str::<Load>(&line).unwrap(), handed);
    }
}
