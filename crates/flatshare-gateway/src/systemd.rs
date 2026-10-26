//! What the gateway shares with systemd: the listening sockets of socket activation (`LISTEN_FDS`), which outlive
//! any one gateway process, and the readiness protocol (`NOTIFY_SOCKET`). On cyan, `flatshare-gateway.socket` holds the
//! sockets and `flatshare-gateway.service` is `Type=notify` (`deploy/cyan/systemd`).

use std::ffi::OsStr;
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::{UnixDatagram, UnixListener};
use std::path::Path;

/// Listening sockets the gateway takes instead of binding its own: systemd's, from socket activation.
/// [`Server::bind_with`](crate::Server::bind_with) matches each to the `[[listen]]` entry of its address, and refuses
/// one that no entry names.
#[derive(Debug, Default)]
pub struct Inherited {
    tcp: Vec<TcpListener>,
    unix: Vec<UnixListener>,
}

impl Inherited {
    /// The sockets passed from fd 3 on, as systemd passes them: `LISTEN_FDS` of them, if `LISTEN_PID` is this
    /// process or unset (listenfd's reading of the protocol). None without `LISTEN_FDS`. It removes both variables
    /// from the environment, so it must run before any other thread starts.
    pub fn from_env() -> io::Result<Self> {
        let mut fds = listenfd::ListenFd::from_env();
        let mut inherited = Self::default();
        for index in 0..fds.len() {
            // A socket of the other kind stays in its place for the second try.
            if let Ok(Some(listener)) = fds.take_tcp_listener(index) {
                inherited.tcp.push(listener);
                continue;
            }
            match fds.take_unix_listener(index) {
                Ok(Some(listener)) => inherited.unix.push(listener),
                Ok(None) => {}
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "LISTEN_FDS: fd {} is neither a TCP nor a Unix stream socket",
                            3 + index
                        ),
                    ));
                }
            }
        }
        Ok(inherited)
    }

    /// Adds a listening TCP socket.
    pub fn tcp(mut self, listener: TcpListener) -> Self {
        self.tcp.push(listener);
        self
    }

    /// Adds a listening Unix socket.
    pub fn unix(mut self, listener: UnixListener) -> Self {
        self.unix.push(listener);
        self
    }

    /// How many sockets there are.
    pub fn len(&self) -> usize {
        self.tcp.len() + self.unix.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The TCP socket bound to `addr`, if one is here; for port 0 (any port), one on its IP address.
    pub(crate) fn take_tcp(&mut self, addr: SocketAddr) -> Option<TcpListener> {
        let found = self.tcp.iter().position(|listener| {
            listener
                .local_addr()
                .is_ok_and(|bound| bound == addr || (addr.port() == 0 && bound.ip() == addr.ip()))
        })?;
        Some(self.tcp.swap_remove(found))
    }

    /// The Unix socket bound to `path`, if one is here.
    pub(crate) fn take_unix(&mut self, path: &Path) -> Option<UnixListener> {
        let found = self.unix.iter().position(|listener| {
            listener
                .local_addr()
                .is_ok_and(|bound| bound.as_pathname() == Some(path))
        })?;
        Some(self.unix.swap_remove(found))
    }

    /// An error naming the sockets that no listener took: systemd would hold them with nobody accepting on them,
    /// so their clients would wait for ever.
    pub(crate) fn none_left(self) -> io::Result<()> {
        let mut left: Vec<String> = self
            .tcp
            .iter()
            .map(|listener| match listener.local_addr() {
                Ok(addr) => format!("tcp {addr}"),
                Err(err) => format!("a TCP socket ({err})"),
            })
            .collect();
        left.extend(self.unix.iter().map(|listener| {
            match listener
                .local_addr()
                .ok()
                .and_then(|addr| addr.as_pathname().map(Path::to_owned))
            {
                Some(path) => format!("unix {}", path.display()),
                None => "an unnamed Unix socket".to_owned(),
            }
        }));
        if left.is_empty() {
            return Ok(());
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "inherited sockets that no [[listen]] entry names, so that nothing would accept on them: {} \
                 (the socket unit's ListenStream= and gateway.toml must name the same addresses)",
                left.join(", ")
            ),
        ))
    }
}

/// Tells the service manager `state`, by sd_notify's protocol (one datagram to `$NOTIFY_SOCKET`, a path or, with a
/// leading `@`, an abstract name): `READY=1` once the gateway is about to serve. Without `NOTIFY_SOCKET`, as when no
/// `Type=notify` unit started it, it does nothing.
pub fn notify(state: &str) -> io::Result<()> {
    let Some(target) = std::env::var_os("NOTIFY_SOCKET") else {
        return Ok(());
    };
    let socket = UnixDatagram::unbound()?;
    match target.as_bytes().strip_prefix(b"@") {
        Some(name) => send_abstract(&socket, name, state),
        None => socket
            .send_to(state.as_bytes(), Path::new(OsStr::new(&target)))
            .map(drop),
    }
}

#[cfg(target_os = "linux")]
fn send_abstract(socket: &UnixDatagram, name: &[u8], state: &str) -> io::Result<()> {
    use std::os::linux::net::SocketAddrExt;
    let addr = std::os::unix::net::SocketAddr::from_abstract_name(name)?;
    socket.send_to_addr(state.as_bytes(), &addr).map(drop)
}

#[cfg(not(target_os = "linux"))]
fn send_abstract(_: &UnixDatagram, _: &[u8], _: &str) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "NOTIFY_SOCKET names an abstract socket, which only Linux has",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inherited_sockets_are_taken_by_their_address_and_the_rest_named() {
        let tcp = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = tcp.local_addr().unwrap();
        let path =
            std::env::temp_dir().join(format!("flatshare-gw-inherit-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let unix = UnixListener::bind(&path).unwrap();
        let other = TcpListener::bind("127.0.0.1:0").unwrap();
        let other_addr = other.local_addr().unwrap();
        let mut inherited = Inherited::default().tcp(tcp).unix(unix).tcp(other);
        assert_eq!(inherited.len(), 3);
        assert!(inherited.take_tcp("127.0.0.2:0".parse().unwrap()).is_none());
        assert!(inherited.take_unix(Path::new("/elsewhere.sock")).is_none());
        assert!(inherited.take_unix(&path).is_some());
        assert!(inherited.take_tcp(addr).is_some());
        assert!(inherited.take_tcp(addr).is_none(), "taken once");
        // Port 0, any port, is the other one on 127.0.0.1.
        let any = inherited.take_tcp("127.0.0.1:0".parse().unwrap()).unwrap();
        assert_eq!(any.local_addr().unwrap(), other_addr);
        inherited = inherited.tcp(any);
        let err = inherited.none_left().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains(&format!(
                "no [[listen]] entry names, so that nothing would accept on them: tcp {other_addr}"
            )),
            "{err}"
        );
        assert!(Inherited::default().none_left().is_ok());
        std::fs::remove_file(&path).unwrap();
    }
}
