//! A reload without a gap: on SIGHUP the gateway starts its successor, from the binary at its own path and with
//! its own arguments (so from the build and the configuration on disk now), with its listening sockets from fd 3
//! on (`LISTEN_FDS`, as systemd passes them) and `LH_GATEWAY_PREDECESSOR` naming it. The successor takes the
//! sockets instead of binding, tells systemd that it is the unit's main process and ready (`MAINPID=`, `READY=1`;
//! the unit's `NotifyAccess=all` lets a process other than the main one say so), and sends SIGTERM to its
//! predecessor, which stops accepting and drains as for a stop. Both accept on the same sockets until then, so no
//! connection is refused or waits. A successor that exits, or is not ready within [`TIMEOUT`], changes nothing:
//! the predecessor serves on.
//!
//! The predecessor watches its successor only until it takes over, and never reaps it after that: once the
//! predecessor exits, systemd, a subreaper, adopts the successor and sees its exit, as a main process's.

use std::io;
use std::os::fd::{OwnedFd, RawFd};
use std::process::{Child, Command, ExitStatus};
use std::time::Duration;

use command_fds::{CommandFdExt, FdMapping};
use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use tokio::signal::unix::{SignalKind, signal};

use crate::systemd;

/// How long a successor has to take over.
pub const TIMEOUT: Duration = Duration::from_secs(10);

/// The environment variable that tells a successor which process started it.
const PREDECESSOR: &str = "LH_GATEWAY_PREDECESSOR";

/// How often the predecessor looks whether its successor exited before taking over.
const POLL: Duration = Duration::from_millis(50);

/// The signals that end a gateway or reload it, installed before anything binds so that an early one still counts.
#[derive(Debug)]
pub struct Signals {
    terminate: tokio::signal::unix::Signal,
    interrupt: tokio::signal::unix::Signal,
    hangup: tokio::signal::unix::Signal,
}

impl Signals {
    /// Installs the handlers of SIGTERM, SIGINT and SIGHUP; it needs a Tokio runtime.
    pub fn install() -> io::Result<Self> {
        Ok(Self {
            terminate: signal(SignalKind::terminate())?,
            interrupt: signal(SignalKind::interrupt())?,
            hangup: signal(SignalKind::hangup())?,
        })
    }
}

/// How a gateway's serving ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Stopped {
    /// SIGTERM or SIGINT with no successor running: a stop.
    Stop,
    /// SIGTERM with a successor running: it took over the sockets, or it is still starting and the unit is being
    /// stopped as a whole. Either way the socket files are not this process's to remove.
    HandedOver,
}

/// The process that started this one by a reload, while it is still this process's parent.
pub fn predecessor() -> Option<u32> {
    let pid: u32 = std::env::var(PREDECESSOR).ok()?.parse().ok()?;
    (std::os::unix::process::parent_id() == pid).then_some(pid)
}

/// Takes over from `predecessor`: tells systemd that this process is the unit's main one and ready, and then
/// relieves the predecessor (SIGTERM), which drains.
pub fn take_over(predecessor: u32) -> io::Result<()> {
    systemd::notify(&format!("MAINPID={}\nREADY=1", std::process::id()))?;
    let pid = i32::try_from(predecessor).map_err(io::Error::other)?;
    kill(Pid::from_raw(pid), Signal::SIGTERM)?;
    Ok(())
}

/// Serves the signals until the gateway is to drain: SIGTERM or SIGINT. Each SIGHUP starts a successor on
/// `listeners` (the gateway's listening sockets, in configuration order) unless one is starting already.
pub async fn until_stopped(mut signals: Signals, listeners: Vec<OwnedFd>) -> Stopped {
    let mut successor: Option<(Child, tokio::time::Instant)> = None;
    let mut poll = tokio::time::interval(POLL);
    loop {
        tokio::select! {
            biased;
            _ = signals.terminate.recv() => break,
            _ = signals.interrupt.recv() => break,
            _ = signals.hangup.recv() => {
                if let Some((child, _)) = &successor {
                    tracing::warn!(successor = child.id(), "reload: a successor is starting already");
                    continue;
                }
                match spawn(&listeners) {
                    Ok(child) => {
                        tracing::info!(successor = child.id(), "reload: started a successor on the listening sockets");
                        successor = Some((child, tokio::time::Instant::now() + TIMEOUT));
                    }
                    Err(err) => tracing::error!("reload failed: cannot start a successor: {err}; serving on"),
                }
            }
            _ = poll.tick(), if successor.is_some() => {
                if let Some((child, deadline)) = &mut successor {
                    if let Some(status) = exited(child) {
                        tracing::error!(
                            successor = child.id(),
                            "reload failed: the successor exited ({status}) before it took over; serving on"
                        );
                        successor = None;
                    } else if tokio::time::Instant::now() >= *deadline {
                        tracing::error!(
                            successor = child.id(),
                            "reload failed: the successor did not take over within {TIMEOUT:?}; stopped it; serving on"
                        );
                        let _ = child.kill();
                        let _ = child.wait();
                        successor = None;
                    }
                }
            }
        }
    }
    match successor {
        // Dropping a std Child neither kills nor reaps it: systemd sees its exit.
        Some((child, _)) => {
            tracing::info!(
                successor = child.id(),
                "stopping with a successor on the sockets"
            );
            Stopped::HandedOver
        }
        None => Stopped::Stop,
    }
}

/// The successor's exit status, if it has exited: it is reaped, as a process that never took over may be.
fn exited(child: &mut Child) -> Option<ExitStatus> {
    child.try_wait().ok().flatten()
}

/// Starts a successor from this process's binary path and arguments, with `listeners` from fd 3 on.
fn spawn(listeners: &[OwnedFd]) -> io::Result<Child> {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .ok_or_else(|| io::Error::other("no argv[0] to start the successor from"))?;
    let mut mappings = Vec::with_capacity(listeners.len());
    for (index, listener) in listeners.iter().enumerate() {
        mappings.push(FdMapping {
            parent_fd: listener.try_clone()?,
            child_fd: 3 + RawFd::try_from(index).map_err(io::Error::other)?,
        });
    }
    let mut command = Command::new(program);
    command
        .args(args)
        .env("LISTEN_FDS", listeners.len().to_string())
        .env_remove("LISTEN_PID")
        .env_remove("LISTEN_FDNAMES")
        .env(PREDECESSOR, std::process::id().to_string())
        .fd_mappings(mappings)
        .map_err(io::Error::other)?;
    command.spawn()
}
