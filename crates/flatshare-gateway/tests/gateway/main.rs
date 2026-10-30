//! The gateway between a test client and in-process mock engines (and, for `[[nodes]]`, mock speech servers), over
//! loopback TCP and a Unix socket.

mod accounting;
mod admission;
mod common;
mod deadlines;
mod drain;
mod events;
mod intake;
mod lifecycle;
mod messages;
mod metrics;
mod nodes;
mod proxy;
mod reload;
mod sockets;
mod streams;
mod upstream;
