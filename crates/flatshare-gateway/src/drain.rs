//! A drain's cut: how a stop or a reload ends the background requests still running `background_drain_s` into its
//! drain.

use std::time::Duration;

use flatshare_proto::SloClass;
use tokio::sync::watch;

/// A drain's cut: the moment, `background_drain_s` into a drain, when the background requests still running are
/// ended. It never comes before a drain.
#[derive(Debug)]
pub(crate) struct Drain {
    cut: watch::Sender<bool>,
    /// `background_drain_s`.
    pub grace: Duration,
}

impl Drain {
    pub fn new(grace: Duration) -> Self {
        Self {
            cut: watch::channel(false).0,
            grace,
        }
    }

    /// What a request of `class` waits on for the cut, which ends background requests only.
    pub fn cut_for(&self, class: Option<SloClass>) -> Option<watch::Receiver<bool>> {
        (class == Some(SloClass::Background)).then(|| self.cut.subscribe())
    }

    pub fn cut(&self) {
        self.cut.send_replace(true);
    }
}

/// Completes at the drain's cut; never for a request the cut does not end (`None`).
pub(crate) async fn cut(cut: &mut Option<watch::Receiver<bool>>) {
    match cut {
        Some(cut) => {
            let _ = cut.wait_for(|cut| *cut).await;
        }
        None => std::future::pending().await,
    }
}

/// Why the drain's cut ended a request, which its client reads in the error.
pub(crate) fn drain_message(grace: Duration) -> String {
    format!(
        "the gateway is stopping or reloading: background requests end {}s into its drain (background_drain_s); \
         send the request again",
        grace.as_secs_f64()
    )
}
