//! `[admission]`: the admission point's policy and the bounds of its queues, validated with the rest of the
//! configuration. Policy "none", the default, configures no admission point.

use std::time::Duration;

use flatshare_core::FROZEN_BOX_POOL_TOKENS;
use flatshare_core::admission::{Ceilings, Lifecycle, Policy, Trigger};
use serde::Deserialize;

use crate::config::seconds;

/// `[admission]`: the policy that decides when a request goes upstream, and the bounds of its queues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    pub policy: Policy,
    /// The requests each class's queue may hold; a request that would wait beyond them gets 429.
    pub queue_limit: usize,
    /// How long a request may wait: `queue_timeout_s`, after which it gets 503, or under the lifecycle policy,
    /// which refuses no request for its wait, `max_defer_s`, after which it is admitted regardless.
    pub max_wait: Duration,
    /// How long a realtime session stays in a state of its turn that no event moves it on from
    /// (`state_timeout_s`). Only the lifecycle policy keeps the sessions' states.
    pub state_timeout: Option<Duration>,
    /// The engine's step budget (`max_num_batched_tokens`), in whose steps the lifecycle policy's throttle prices a
    /// background request's prefill; `None` without the throttle.
    pub step_tokens: Option<u64>,
}

/// `[admission]` as written. Each key but `policy` belongs to some of the policies, and setting it for another
/// one is an error, so no setting silently does nothing; nor, under the lifecycle policy, is a key that no
/// mechanism switched on reads.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct RawAdmission {
    policy: PolicyName,
    background_max_in_flight: Option<u32>,
    pool_tokens: Option<u64>,
    reserved_tokens: Option<u64>,
    ceilings: Option<RawCeilings>,
    headroom: Option<bool>,
    headroom_tokens: Option<u64>,
    trigger: Option<TriggerName>,
    throttle: Option<bool>,
    critical_prefill_budget_ms: Option<f64>,
    max_num_batched_tokens: Option<u64>,
    degrade: Option<bool>,
    degrade_max_tokens: Option<u32>,
    max_defer_s: Option<f64>,
    state_timeout_s: Option<f64>,
    queue_limit: Option<usize>,
    queue_timeout_s: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCeilings {
    realtime: u32,
    interactive: u32,
    background: u32,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum PolicyName {
    #[default]
    None,
    StaticCap,
    TokenReservation,
    FlowControl,
    PreemptResubmit,
    /// The lifecycle policy (`policy = "lifecycle"`; arm files from before the project's rename, and their
    /// records, spell it `leaseholder`).
    #[serde(alias = "leaseholder")]
    Lifecycle,
}

impl PolicyName {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::StaticCap => "static-cap",
            Self::TokenReservation => "token-reservation",
            Self::FlowControl => "flow-control",
            Self::PreemptResubmit => "preempt-resubmit",
            Self::Lifecycle => "lifecycle",
        }
    }
}

/// `trigger`: what arms the lifecycle policy's headroom.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum TriggerName {
    Onset,
    Request,
}

impl RawAdmission {
    /// The admission point the table configures; `None` under policy "none".
    pub(crate) fn validate(self) -> Result<Option<Admission>, String> {
        let name = self.policy.as_str();
        let reads: &[&str] = match self.policy {
            PolicyName::None => &[],
            PolicyName::StaticCap => {
                &["background_max_in_flight", "queue_limit", "queue_timeout_s"]
            }
            PolicyName::TokenReservation => &[
                "pool_tokens",
                "reserved_tokens",
                "queue_limit",
                "queue_timeout_s",
            ],
            PolicyName::FlowControl => &["ceilings", "queue_limit", "queue_timeout_s"],
            PolicyName::PreemptResubmit => &["queue_limit", "queue_timeout_s"],
            PolicyName::Lifecycle => &[
                "pool_tokens",
                "headroom",
                "headroom_tokens",
                "trigger",
                "throttle",
                "critical_prefill_budget_ms",
                "max_num_batched_tokens",
                "degrade",
                "degrade_max_tokens",
                "max_defer_s",
                "state_timeout_s",
                "queue_limit",
            ],
        };
        let set = [
            (
                "background_max_in_flight",
                self.background_max_in_flight.is_some(),
            ),
            ("pool_tokens", self.pool_tokens.is_some()),
            ("reserved_tokens", self.reserved_tokens.is_some()),
            ("ceilings", self.ceilings.is_some()),
            ("headroom", self.headroom.is_some()),
            ("headroom_tokens", self.headroom_tokens.is_some()),
            ("trigger", self.trigger.is_some()),
            ("throttle", self.throttle.is_some()),
            (
                "critical_prefill_budget_ms",
                self.critical_prefill_budget_ms.is_some(),
            ),
            (
                "max_num_batched_tokens",
                self.max_num_batched_tokens.is_some(),
            ),
            ("degrade", self.degrade.is_some()),
            ("degrade_max_tokens", self.degrade_max_tokens.is_some()),
            ("max_defer_s", self.max_defer_s.is_some()),
            ("state_timeout_s", self.state_timeout_s.is_some()),
            ("queue_limit", self.queue_limit.is_some()),
            ("queue_timeout_s", self.queue_timeout_s.is_some()),
        ];
        if let Some((key, _)) = set.iter().find(|(key, set)| *set && !reads.contains(key)) {
            return Err(format!("admission.{key} does not apply to policy {name}"));
        }
        // SMG's bulk class queues 1,024 requests.
        let queue_limit = self.queue_limit.unwrap_or(1024);
        if let PolicyName::Lifecycle = self.policy {
            return self.lifecycle(queue_limit).map(Some);
        }
        let needs = |key: &str| format!("admission: policy {name} needs {key}");
        let policy = match self.policy {
            PolicyName::None => return Ok(None),
            PolicyName::StaticCap => {
                let cap = self
                    .background_max_in_flight
                    .ok_or_else(|| needs("background_max_in_flight"))?;
                if cap == 0 {
                    return Err("admission.background_max_in_flight must be positive".into());
                }
                Policy::StaticCap {
                    background_max_in_flight: cap,
                }
            }
            PolicyName::TokenReservation => {
                let pool_tokens = self.pool_tokens.unwrap_or(FROZEN_BOX_POOL_TOKENS);
                let reserved_tokens = self
                    .reserved_tokens
                    .ok_or_else(|| needs("reserved_tokens"))?;
                if reserved_tokens >= pool_tokens {
                    return Err(format!(
                        "admission.reserved_tokens ({reserved_tokens}) must be less than pool_tokens \
                         ({pool_tokens})"
                    ));
                }
                Policy::TokenReservation {
                    pool_tokens,
                    reserved_tokens,
                }
            }
            PolicyName::FlowControl => {
                let RawCeilings {
                    realtime,
                    interactive,
                    background,
                } = self.ceilings.ok_or_else(|| needs("ceilings"))?;
                if background == 0 {
                    return Err("admission.ceilings must be positive".into());
                }
                if realtime < interactive || interactive < background {
                    return Err(
                        "admission.ceilings must not increase from realtime to background: the lower \
                         classes are held back first"
                            .into(),
                    );
                }
                Policy::FlowControl {
                    ceilings: Ceilings {
                        realtime,
                        interactive,
                        background,
                    },
                }
            }
            PolicyName::PreemptResubmit => Policy::PreemptResubmit,
            PolicyName::Lifecycle => unreachable!("validated above"),
        };
        Ok(Some(Admission {
            policy,
            queue_limit,
            // A minute ends a wait well before clients give up on their own (AIPerf's replay after 300 s,
            // OpenAI's SDK after 600 s).
            max_wait: seconds(
                "admission.queue_timeout_s",
                self.queue_timeout_s.unwrap_or(60.0),
            )?,
            state_timeout: None,
            step_tokens: None,
        }))
    }

    /// The lifecycle policy. Each mechanism is on unless switched off and needs its own keys while it is on, and a
    /// key may be set only while a mechanism that reads it is on.
    fn lifecycle(self, queue_limit: usize) -> Result<Admission, String> {
        let headroom = self.headroom.unwrap_or(true);
        let throttle = self.throttle.unwrap_or(true);
        let degrade = self.degrade.unwrap_or(true);
        // The headroom and degrade read the pool and a turn's headroom; the mechanisms that hold requests back,
        // the headroom and the throttle, read the queue and how long a request may be deferred.
        let sizing = headroom || degrade;
        let holding = headroom || throttle;
        for (key, set, on, off) in [
            (
                "trigger",
                self.trigger.is_some(),
                headroom,
                "headroom = false",
            ),
            (
                "critical_prefill_budget_ms",
                self.critical_prefill_budget_ms.is_some(),
                throttle,
                "throttle = false",
            ),
            (
                "max_num_batched_tokens",
                self.max_num_batched_tokens.is_some(),
                throttle,
                "throttle = false",
            ),
            (
                "degrade_max_tokens",
                self.degrade_max_tokens.is_some(),
                degrade,
                "degrade = false",
            ),
            (
                "headroom_tokens",
                self.headroom_tokens.is_some(),
                sizing,
                "headroom = false and degrade = false",
            ),
            (
                "pool_tokens",
                self.pool_tokens.is_some(),
                sizing,
                "headroom = false and degrade = false",
            ),
            (
                "max_defer_s",
                self.max_defer_s.is_some(),
                holding,
                "headroom = false and throttle = false",
            ),
            (
                "queue_limit",
                self.queue_limit.is_some(),
                holding,
                "headroom = false and throttle = false",
            ),
        ] {
            if set && !on {
                return Err(format!("admission.{key} does not apply with {off}"));
            }
        }
        let needs =
            |key: &str, off: &str| format!("admission: policy lifecycle needs {key}, unless {off}");
        let pool_tokens = self.pool_tokens.unwrap_or(FROZEN_BOX_POOL_TOKENS);
        let headroom_tokens = match self.headroom_tokens {
            Some(tokens) if (1..pool_tokens).contains(&tokens) => tokens,
            Some(tokens) => {
                return Err(format!(
                    "admission.headroom_tokens ({tokens}) must be positive and less than pool_tokens \
                     ({pool_tokens})"
                ));
            }
            None if sizing => {
                return Err(needs(
                    "headroom_tokens",
                    "headroom = false and degrade = false",
                ));
            }
            None => 0,
        };
        let critical_prefill_budget = match self.critical_prefill_budget_ms {
            Some(ms) => Some(Duration::try_from_secs_f64(ms / 1e3).map_err(|_| {
                "admission.critical_prefill_budget_ms must be a number of milliseconds, 0 or more"
            })?),
            None if throttle => {
                return Err(needs("critical_prefill_budget_ms", "throttle = false"));
            }
            None => None,
        };
        // The frozen box's `--max-num-batched-tokens`.
        let step_tokens = match self.max_num_batched_tokens {
            Some(0) => return Err("admission.max_num_batched_tokens must be positive".into()),
            tokens => throttle.then(|| tokens.unwrap_or(512)),
        };
        let degrade_max_tokens = match self.degrade_max_tokens {
            Some(0) => return Err("admission.degrade_max_tokens must be positive".into()),
            None if degrade => return Err(needs("degrade_max_tokens", "degrade = false")),
            cap => cap,
        };
        let trigger = match self.trigger {
            None | Some(TriggerName::Onset) => Trigger::Onset,
            Some(TriggerName::Request) => Trigger::Request,
        };
        let lifecycle = Lifecycle {
            pool_tokens,
            headroom_tokens,
            headroom: headroom.then_some(trigger),
            critical_prefill_budget,
            degrade_max_tokens,
        };
        // Longer than an E2 caller's turn is armed (about 4 s from speech onset to the reply's first audio), so
        // aging admits only what a stuck or unbroken arming holds back.
        let max_defer_s = self.max_defer_s.unwrap_or(10.0);
        // Longer than any state of an E2 caller's turn lasts (lines of 2 to 3 s, replies of a few seconds of
        // audio), so only a lost event ends a state by timeout.
        let state_timeout_s = self.state_timeout_s.unwrap_or(10.0);
        Ok(Admission {
            policy: Policy::Lifecycle(lifecycle),
            queue_limit,
            max_wait: seconds("admission.max_defer_s", max_defer_s)?,
            state_timeout: Some(seconds("admission.state_timeout_s", state_timeout_s)?),
            step_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[admission]` from its keys, as the configuration file holds them.
    fn parse(keys: &str) -> Result<Option<Admission>, String> {
        toml::from_str::<RawAdmission>(keys)
            .map_err(|e| e.to_string())?
            .validate()
    }

    #[test]
    fn admission_policies_take_their_own_keys_and_defaults() {
        let admission = parse;
        assert_eq!(admission("").unwrap(), None);
        assert_eq!(admission("policy = \"none\"").unwrap(), None);
        let cap = admission("policy = \"static-cap\"\nbackground_max_in_flight = 4").unwrap();
        assert_eq!(
            cap,
            Some(Admission {
                policy: Policy::StaticCap {
                    background_max_in_flight: 4
                },
                queue_limit: 1024,
                max_wait: Duration::from_secs(60),
                state_timeout: None,
                step_tokens: None,
            })
        );
        let reservation = admission(
            "policy = \"token-reservation\"\nreserved_tokens = 2048\nqueue_limit = 8\nqueue_timeout_s = 290",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            reservation.policy,
            Policy::TokenReservation {
                pool_tokens: FROZEN_BOX_POOL_TOKENS,
                reserved_tokens: 2048
            }
        );
        assert_eq!(
            (reservation.queue_limit, reservation.max_wait),
            (8, Duration::from_secs(290))
        );
        let flow = admission(
            "policy = \"flow-control\"\nceilings = { realtime = 16, interactive = 12, background = 8 }",
        )
        .unwrap()
        .unwrap();
        let ceilings = Ceilings {
            realtime: 16,
            interactive: 12,
            background: 8,
        };
        assert_eq!(flow.policy, Policy::FlowControl { ceilings });
        let preempt = admission("policy = \"preempt-resubmit\"").unwrap();
        assert_eq!(preempt.map(|a| a.policy), Some(Policy::PreemptResubmit));

        for (keys, cause) in [
            ("policy = \"static_cap\"", "unknown variant"),
            (
                "policy = \"none\"\nqueue_limit = 4",
                "queue_limit does not apply to policy none",
            ),
            (
                "policy = \"static-cap\"",
                "static-cap needs background_max_in_flight",
            ),
            (
                "policy = \"static-cap\"\nbackground_max_in_flight = 0",
                "must be positive",
            ),
            (
                "policy = \"static-cap\"\nbackground_max_in_flight = 2\nreserved_tokens = 1",
                "reserved_tokens does not apply to policy static-cap",
            ),
            (
                "policy = \"static-cap\"\nbackground_max_in_flight = 2\nqueue_timeout_s = 0",
                "queue_timeout_s must be a positive",
            ),
            ("policy = \"token-reservation\"", "needs reserved_tokens"),
            (
                "policy = \"token-reservation\"\nreserved_tokens = 40000",
                "must be less than pool_tokens (36400)",
            ),
            ("polcy = \"none\"", "unknown field"),
            ("policy = \"flow-control\"", "flow-control needs ceilings"),
            (
                "policy = \"flow-control\"\nceilings = { realtime = 8, background = 4 }",
                "missing field `interactive`",
            ),
            (
                "policy = \"flow-control\"\nceilings = { realtime = 4, interactive = 8, background = 4 }",
                "must not increase",
            ),
            (
                "policy = \"flow-control\"\nceilings = { realtime = 4, interactive = 4, background = 0 }",
                "must be positive",
            ),
            (
                "policy = \"preempt-resubmit\"\nbackground_max_in_flight = 4",
                "does not apply to policy preempt-resubmit",
            ),
        ] {
            let err = admission(keys).unwrap_err();
            assert!(err.contains(cause), "{keys}: {err}");
        }
    }

    #[test]
    fn flatshare_is_the_lifecycle_policys_name_in_e2s_arm_files() {
        let keys =
            "headroom_tokens = 1024\ncritical_prefill_budget_ms = 150\ndegrade_max_tokens = 16";
        let lifecycle = parse(&format!("policy = \"lifecycle\"\n{keys}")).unwrap();
        assert!(matches!(
            lifecycle,
            Some(Admission {
                policy: Policy::Lifecycle(_),
                ..
            })
        ));
        assert_eq!(
            parse(&format!("policy = \"leaseholder\"\n{keys}")).unwrap(),
            lifecycle
        );
    }

    #[test]
    fn the_lifecycle_policy_switches_each_mechanism_with_its_own_keys() {
        let admission =
            |keys: &str| parse(&format!("policy = \"lifecycle\"\n{keys}")).map(Option::unwrap);
        let all =
            "headroom_tokens = 1024\ncritical_prefill_budget_ms = 250\ndegrade_max_tokens = 24";
        let arm9 = admission(all).unwrap();
        let full = Lifecycle {
            pool_tokens: FROZEN_BOX_POOL_TOKENS,
            headroom_tokens: 1024,
            headroom: Some(Trigger::Onset),
            critical_prefill_budget: Some(Duration::from_millis(250)),
            degrade_max_tokens: Some(24),
        };
        assert_eq!(
            arm9,
            Admission {
                policy: Policy::Lifecycle(full),
                queue_limit: 1024,
                max_wait: Duration::from_secs(10),
                state_timeout: Some(Duration::from_secs(10)),
                step_tokens: Some(512),
            }
        );
        let tuned = admission(&format!(
            "{all}\ntrigger = \"request\"\npool_tokens = 20000\nmax_defer_s = 4.5\nstate_timeout_s = 30\n\
             queue_limit = 16\nmax_num_batched_tokens = 1024"
        ))
        .unwrap();
        let request = Lifecycle {
            pool_tokens: 20_000,
            headroom: Some(Trigger::Request),
            ..full
        };
        assert_eq!(tuned.policy, Policy::Lifecycle(request));
        assert_eq!(
            (tuned.queue_limit, tuned.max_wait, tuned.state_timeout),
            (
                16,
                Duration::from_secs_f64(4.5),
                Some(Duration::from_secs(30))
            )
        );
        assert_eq!(tuned.step_tokens, Some(1024));

        // The ablations: each mechanism off, without its keys.
        let ablation = |keys: &str| match admission(keys).unwrap().policy {
            Policy::Lifecycle(lifecycle) => lifecycle,
            other => panic!("{other:?}"),
        };
        let no_throttle = "headroom_tokens = 1024\ndegrade_max_tokens = 24\nthrottle = false";
        assert_eq!(
            ablation(no_throttle),
            Lifecycle {
                critical_prefill_budget: None,
                ..full
            }
        );
        assert_eq!(admission(no_throttle).unwrap().step_tokens, None);
        let no_headroom = "headroom = false\nheadroom_tokens = 1024\ncritical_prefill_budget_ms = 250\n\
                           degrade_max_tokens = 24";
        assert_eq!(
            ablation(no_headroom),
            Lifecycle {
                headroom: None,
                ..full
            }
        );
        let no_degrade =
            "headroom_tokens = 1024\ncritical_prefill_budget_ms = 250\ndegrade = false";
        assert_eq!(
            ablation(no_degrade),
            Lifecycle {
                degrade_max_tokens: None,
                ..full
            }
        );
        let throttle_alone = "headroom = false\ndegrade = false\ncritical_prefill_budget_ms = 0";
        assert_eq!(
            ablation(throttle_alone),
            Lifecycle {
                headroom_tokens: 0,
                headroom: None,
                critical_prefill_budget: Some(Duration::ZERO),
                degrade_max_tokens: None,
                ..full
            }
        );

        for (keys, cause) in [
            (
                "",
                "lifecycle needs headroom_tokens, unless headroom = false and degrade = false",
            ),
            (
                "headroom_tokens = 1024\ndegrade_max_tokens = 24",
                "needs critical_prefill_budget_ms, unless throttle = false",
            ),
            (
                "headroom_tokens = 1024\ncritical_prefill_budget_ms = 250",
                "needs degrade_max_tokens, unless degrade = false",
            ),
            (
                &format!("{all}\nthrottle = false"),
                "critical_prefill_budget_ms does not apply with throttle = false",
            ),
            (
                &format!("{all}\ndegrade = false"),
                "degrade_max_tokens does not apply with degrade = false",
            ),
            (
                "headroom_tokens = 1024\ndegrade_max_tokens = 24\nthrottle = false\nmax_num_batched_tokens = 512",
                "max_num_batched_tokens does not apply with throttle = false",
            ),
            (
                &format!("{all}\nmax_num_batched_tokens = 0"),
                "max_num_batched_tokens must be positive",
            ),
            (
                &format!("{all}\nheadroom = false\ntrigger = \"onset\""),
                "trigger does not apply with headroom = false",
            ),
            (
                "headroom = false\ndegrade = false\nheadroom_tokens = 1\ncritical_prefill_budget_ms = 1",
                "headroom_tokens does not apply with headroom = false and degrade = false",
            ),
            (
                &format!("{all}\nqueue_timeout_s = 60"),
                "queue_timeout_s does not apply to policy lifecycle",
            ),
            (
                "headroom_tokens = 36400\ncritical_prefill_budget_ms = 250\ndegrade_max_tokens = 24",
                "must be positive and less than pool_tokens (36400)",
            ),
            (
                "headroom_tokens = 1024\ncritical_prefill_budget_ms = -1\ndegrade_max_tokens = 24",
                "a number of milliseconds, 0 or more",
            ),
            (
                "headroom_tokens = 1024\ncritical_prefill_budget_ms = 250\ndegrade_max_tokens = 0",
                "degrade_max_tokens must be positive",
            ),
            (
                &format!("{all}\nmax_defer_s = 0"),
                "max_defer_s must be a positive",
            ),
            (&format!("{all}\ntrigger = \"speech\""), "unknown variant"),
        ] {
            let err = admission(keys).unwrap_err();
            assert!(err.contains(cause), "{keys}: {err}");
        }
        let err = parse("policy = \"static-cap\"\nbackground_max_in_flight = 1\nmax_defer_s = 1")
            .unwrap_err();
        assert!(
            err.contains("max_defer_s does not apply to policy static-cap"),
            "{err}"
        );
    }

    #[test]
    fn a_lifecycle_key_applies_only_while_a_mechanism_that_reads_it_is_on() {
        let admission =
            |keys: &str| parse(&format!("policy = \"lifecycle\"\n{keys}")).map(Option::unwrap);
        // The throttle alone reads no pool; degrade alone defers nothing, so ages nothing and queues nothing.
        let throttle_alone = "headroom = false\ndegrade = false\ncritical_prefill_budget_ms = 100";
        let degrade_alone =
            "headroom = false\nthrottle = false\nheadroom_tokens = 200\ndegrade_max_tokens = 16";
        for (keys, cause) in [
            (
                format!("{throttle_alone}\npool_tokens = 1000"),
                "pool_tokens does not apply with headroom = false and degrade = false",
            ),
            (
                format!("{degrade_alone}\nmax_defer_s = 5"),
                "max_defer_s does not apply with headroom = false and throttle = false",
            ),
            (
                format!("{degrade_alone}\nqueue_limit = 16"),
                "queue_limit does not apply with headroom = false and throttle = false",
            ),
        ] {
            let err = admission(&keys).unwrap_err();
            assert!(err.contains(cause), "{keys}: {err}");
        }
        // Each is read while one mechanism that reads it is on, and the sessions are followed with every one off.
        for keys in [
            format!("{degrade_alone}\npool_tokens = 1000"),
            format!("{throttle_alone}\nmax_defer_s = 5\nqueue_limit = 16"),
            "headroom = false\nthrottle = false\ndegrade = false\nstate_timeout_s = 30".to_owned(),
        ] {
            assert!(admission(&keys).is_ok(), "{keys}");
        }
    }
}
