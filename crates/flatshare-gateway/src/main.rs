//! `flatshare-gateway --config <file>`: serve until SIGTERM or SIGINT, then drain in-flight requests and exit 0. SIGHUP
//! reloads: a successor, from the binary and the configuration on disk, takes the listening sockets over, and this
//! process drains (`flatshare_gateway::handover`). The listening sockets systemd passes (`LISTEN_FDS`, socket
//! activation) are served instead of binding new ones, and `READY=1` goes to `NOTIFY_SOCKET` once the gateway is about
//! to serve. A configuration error exits 2; a listener that cannot bind, or an inherited socket that no listener names,
//! exits 1.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use flatshare_gateway::config::{LogFormat, LogLevel};
use flatshare_gateway::handover::{self, Signals, Stopped};
use flatshare_gateway::{Config, Inherited, LogSink, Server, systemd};
use tracing_subscriber::filter::LevelFilter;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Flatshare gateway: the serving control plane in front of vLLM"
)]
struct Args {
    /// The configuration file (TOML).
    #[arg(long)]
    config: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config = match Config::load(&args.config) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("flatshare-gateway: {err}");
            return ExitCode::from(2);
        }
    };
    init_logging(config.log_format, config.log_level);
    // Before the runtime's threads: reading LISTEN_FDS removes it from the environment.
    let inherited = match Inherited::from_env() {
        Ok(inherited) => inherited,
        Err(err) => {
            tracing::error!(error = %err, "cannot take the inherited sockets");
            return ExitCode::FAILURE;
        }
    };
    let predecessor = handover::predecessor();
    let ledger_path = predecessor.and(handover::ledger());
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            tracing::error!(error = %err, "cannot start the runtime");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(async {
        // Signal handlers go in before anything binds, so an early SIGTERM still drains.
        let signals = match Signals::install() {
            Ok(signals) => signals,
            Err(err) => {
                tracing::error!(error = %err, "cannot install signal handlers");
                return ExitCode::FAILURE;
            }
        };
        let server = match Server::bind_with(config, Arc::new(LogSink), inherited).await {
            Ok(server) => server,
            Err(err) => {
                tracing::error!(error = %err, "cannot listen");
                return ExitCode::FAILURE;
            }
        };
        // A reload's successor counts what its predecessor holds on the engine until the predecessor exits.
        let followed = match &ledger_path {
            Some(path) => match server.follow(path).await {
                Ok(followed) => Some(followed),
                Err(err) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %err,
                        "cannot read the predecessor's ledger; its requests will not count"
                    );
                    None
                }
            },
            None => None,
        };
        // A reload's successor gets copies of the listening sockets.
        let listeners = match server.listener_fds() {
            Ok(listeners) => listeners,
            Err(err) => {
                tracing::error!(error = %err, "cannot keep the listening sockets for a reload");
                return ExitCode::FAILURE;
            }
        };
        // The listeners queue connections from here on, and run() accepts at once.
        let ready = match predecessor {
            Some(pid) => handover::take_over(pid)
                .inspect(|()| tracing::info!(predecessor = pid, "took over the listening sockets")),
            None => systemd::notify("READY=1"),
        };
        if let Err(err) = ready {
            tracing::warn!(error = %err, "cannot say that the gateway is ready");
        }
        // The predecessor admits nothing more once it reads the SIGTERM above; until it says so, both would admit.
        if let Some(mut followed) = followed
            && tokio::time::timeout(handover::CLOSE_WAIT, followed.closed())
                .await
                .is_err()
        {
            tracing::warn!(
                wait_s = handover::CLOSE_WAIT.as_secs_f64(),
                "the predecessor did not say in time that it admits nothing more; serving anyway"
            );
        }
        let handed_over = server.handover();
        let ledger = server.ledger();
        let shutdown = async move {
            if handover::until_stopped(signals, listeners, ledger).await == Stopped::HandedOver {
                handed_over.done();
            }
        };
        server.run(shutdown).await;
        ExitCode::SUCCESS
    })
}

fn init_logging(format: LogFormat, level: LogLevel) {
    let level = match level {
        LogLevel::Error => LevelFilter::ERROR,
        LogLevel::Warn => LevelFilter::WARN,
        LogLevel::Info => LevelFilter::INFO,
        LogLevel::Debug => LevelFilter::DEBUG,
        LogLevel::Trace => LevelFilter::TRACE,
    };
    let builder = tracing_subscriber::fmt().with_max_level(level);
    match format {
        LogFormat::Text => builder.init(),
        LogFormat::Json => builder.json().flatten_event(true).init(),
    }
}
