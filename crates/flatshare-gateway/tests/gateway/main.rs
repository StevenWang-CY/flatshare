//! The gateway between a test client and an in-process mock engine, over loopback TCP and a Unix socket.

mod accounting;
mod admission;
mod common;
mod deadlines;
mod drain;
mod events;
mod intake;
mod lifecycle;
mod metrics;
mod proxy;
mod reload;
mod sockets;
mod streams;
mod upstream;
