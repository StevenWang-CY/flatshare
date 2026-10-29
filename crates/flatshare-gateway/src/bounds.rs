//! What a request may make the engine generate: how many tokens its reply may take, and how many choices it may ask
//! for. The gateway reads both as the route's engine reads them, so that no body it forwards means more to the engine
//! than the gateway allowed.
//!
//! vLLM 0.32's request models allow fields they do not know, and each route reads its own limit: Chat Completions
//! `max_completion_tokens` and, when that is null or absent, its deprecated `max_tokens`; Completions `max_tokens`
//! alone, 16 when absent and no limit when null; the Responses API `max_output_tokens` alone. A field of another route
//! is extra, and ignored. A capped body is therefore rewritten to its route's own fields: Chat Completions gets both of
//! its fields at the same value, Completions `max_tokens` and Responses `max_output_tokens`, each the lower of the
//! client's limit and the cap, and the fields the route does not read are removed. Python's `json` keeps the last of
//! a key given twice, which the gateway's own parse would not, so a body that sets one of these fields twice is
//! refused, as is a value that is not null or a whole number (vLLM's pydantic models would read `"64"` or `64.0` as a
//! count, which the gateway does not try to reproduce).
//!
//! `n` is a request's choices, and with `use_beam_search` its beam's width; `best_of`, which vLLM 0.32 no longer reads,
//! is the same multiplier for engines that do. Both are bounded by `max_choices`.

use std::fmt;
use std::io::Write as _;

use serde_json::value::RawValue;

use crate::metrics::Route;

/// The body fields that limit a reply's tokens, which a capped body has rewritten to its route's own.
pub(crate) const REPLY_LIMITS: [&str; 3] =
    ["max_completion_tokens", "max_tokens", "max_output_tokens"];

/// The body fields that multiply a request's work: its choices (and a beam search's width), and `best_of`.
pub(crate) const CHOICES: [&str; 2] = ["n", "best_of"];

/// vLLM's reply limit for a Completions body without `max_tokens` (its request model's default).
pub(crate) const COMPLETIONS_DEFAULT_MAX_TOKENS: u64 = 16;

/// One reply-limit field of a body, as the engine reads it: absent, null, or a count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Limit {
    #[default]
    Absent,
    Null,
    Count(u64),
}

impl Limit {
    fn count(self) -> Option<u64> {
        match self {
            Self::Count(count) => Some(count),
            Self::Absent | Self::Null => None,
        }
    }
}

/// What a body asks the engine for: its reply-limit fields, in [`REPLY_LIMITS`] order, validated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Asked {
    limits: [Limit; 3],
}

impl Asked {
    /// The body's reply limit as the engine of `route` reads it, in tokens; `None` for no limit.
    pub fn limit(&self, route: Route) -> Option<u64> {
        let [completion, tokens, output] = self.limits;
        match route {
            Route::ChatCompletions => completion.count().or(tokens.count()),
            Route::Completions => match tokens {
                Limit::Absent => Some(COMPLETIONS_DEFAULT_MAX_TOKENS),
                Limit::Null => None,
                Limit::Count(count) => Some(count),
            },
            Route::Responses => output.count(),
            Route::Models | Route::Events | Route::Sessions | Route::Session | Route::Other => None,
        }
    }

    /// The limit a body capped at `cap` gets: the lower of its own and the cap.
    pub fn capped(&self, route: Route, cap: u32) -> u64 {
        let cap = u64::from(cap);
        self.limit(route).map_or(cap, |asked| asked.min(cap))
    }
}

/// Why a body's limits or choices were refused (400).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Invalid {
    /// One of the fields this module reads is set twice.
    Duplicate(&'static str),
    /// A reply limit that is not null or a whole number.
    NotACount(&'static str),
    /// `n` or `best_of` not null or a whole number from 1.
    NotAChoice(&'static str),
    /// `n` or `best_of` above `max_choices`.
    TooManyChoices {
        field: &'static str,
        asked: u64,
        max: u32,
    },
}

impl Invalid {
    /// The field at fault, the error's `param`.
    pub fn field(&self) -> &'static str {
        match self {
            Self::Duplicate(field) | Self::NotACount(field) | Self::NotAChoice(field) => field,
            Self::TooManyChoices { field, .. } => field,
        }
    }

    /// The error's `code`.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Duplicate(_) => "duplicate_field",
            Self::NotACount(_) => "invalid_reply_limit",
            Self::NotAChoice(_) => "invalid_choices",
            Self::TooManyChoices { .. } => "too_many_choices",
        }
    }
}

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Duplicate(field) => write!(f, "the request body sets {field} more than once"),
            Self::NotACount(field) => write!(
                f,
                "{field} must be null or a whole number of tokens from 0 to {}",
                u64::MAX
            ),
            Self::NotAChoice(field) => write!(f, "{field} must be null or a whole number from 1"),
            Self::TooManyChoices { field, asked, max } => write!(
                f,
                "{field} is {asked}, and this gateway serves at most {max} (max_choices) a request"
            ),
        }
    }
}

/// Reads and checks the reply limits and choices of a body's top-level members: each field at most once, every
/// limit null or a count, every choice null or a count from 1 to `max_choices`.
pub(crate) fn read<'a>(
    members: impl IntoIterator<Item = (&'a str, &'a RawValue)>,
    max_choices: u32,
) -> Result<Asked, Invalid> {
    let mut asked = Asked::default();
    let mut seen = [false; REPLY_LIMITS.len() + CHOICES.len()];
    for (key, value) in members {
        if let Some(i) = REPLY_LIMITS.iter().position(|field| *field == key) {
            let field = REPLY_LIMITS[i];
            if std::mem::replace(&mut seen[i], true) {
                return Err(Invalid::Duplicate(field));
            }
            asked.limits[i] = match count(value) {
                Some(Some(count)) => Limit::Count(count),
                Some(None) => Limit::Null,
                None => return Err(Invalid::NotACount(field)),
            };
        } else if let Some(i) = CHOICES.iter().position(|field| *field == key) {
            let field = CHOICES[i];
            if std::mem::replace(&mut seen[REPLY_LIMITS.len() + i], true) {
                return Err(Invalid::Duplicate(field));
            }
            match count(value) {
                Some(None) => {}
                Some(Some(0)) | None => return Err(Invalid::NotAChoice(field)),
                Some(Some(asked)) if asked > u64::from(max_choices) => {
                    return Err(Invalid::TooManyChoices {
                        field,
                        asked,
                        max: max_choices,
                    });
                }
                Some(Some(_)) => {}
            }
        }
    }
    Ok(asked)
}

/// A value that is null (`Some(None)`) or a whole number that fits 64 bits (`Some(Some(n))`); `None` for anything
/// else: a string, a fraction or an exponent, a negative number, a boolean, an array or an object.
fn count(value: &RawValue) -> Option<Option<u64>> {
    if value.get() == "null" {
        return Some(None);
    }
    serde_json::from_str::<u64>(value.get()).ok().map(Some)
}

/// Whether `key` is one of the reply limits a capped body has rewritten.
pub(crate) fn is_reply_limit(key: &str) -> bool {
    REPLY_LIMITS.contains(&key)
}

/// Writes `route`'s own reply limits at `limit`, each followed by a comma: Chat Completions' both, at the same value,
/// so that the engine reads the same limit whichever it takes; Completions' `max_tokens`; Responses'
/// `max_output_tokens`.
pub(crate) fn write_limit(json: &mut Vec<u8>, route: Route, limit: u64) {
    let fields: &[&str] = match route {
        Route::ChatCompletions => &["max_completion_tokens", "max_tokens"],
        Route::Completions => &["max_tokens"],
        Route::Responses => &["max_output_tokens"],
        Route::Models | Route::Events | Route::Sessions | Route::Session | Route::Other => &[],
    };
    for field in fields {
        write!(json, "\"{field}\":{limit},").expect("writing to a Vec cannot fail");
    }
}

/// The tests: each route's limit read as vLLM reads it, the cap at the lower of the two, each route's own fields
/// written; every field twice, every value that is not a count, `n` and `best_of` over `max_choices` and under 1
/// refused; a cap in the rewrite equal byte for byte to the cap of the rewritten body, and degrade's over a class's
/// keeping the lower; the fixture of 285 cases current.
#[cfg(test)]
mod tests {
    use super::*;

    /// Every member in order, duplicates kept, as the gateway's own parse reads a body.
    fn members(body: &str) -> Vec<(String, Box<RawValue>)> {
        let crate::sanitize::TopLevel(fields) = serde_json::from_str(body).unwrap();
        fields
            .into_iter()
            .map(|(key, value)| (key, value.to_owned()))
            .collect()
    }

    fn asked(body: &str, max_choices: u32) -> Result<Asked, Invalid> {
        let members = members(body);
        read(members.iter().map(|(k, v)| (k.as_str(), &**v)), max_choices)
    }

    #[test]
    fn each_route_reads_its_own_limit_as_vllm_does() {
        use Route::*;
        for (body, chat, completions, responses) in [
            (r#"{}"#, None, Some(16), None),
            (r#"{"max_tokens":40}"#, Some(40), Some(40), None),
            (r#"{"max_completion_tokens":30}"#, Some(30), Some(16), None),
            (r#"{"max_output_tokens":20}"#, None, Some(16), Some(20)),
            // Chat Completions reads max_completion_tokens first, whatever max_tokens says.
            (
                r#"{"max_tokens":40,"max_completion_tokens":30}"#,
                Some(30),
                Some(40),
                None,
            ),
            (
                r#"{"max_tokens":40,"max_completion_tokens":null}"#,
                Some(40),
                Some(40),
                None,
            ),
            // A null limit is no limit; on Completions, unlike an absent one, which is 16.
            (r#"{"max_tokens":null}"#, None, None, None),
            (r#"{"max_output_tokens":null}"#, None, Some(16), None),
            (r#"{"max_tokens":0}"#, Some(0), Some(0), None),
        ] {
            let asked = asked(body, 1).unwrap();
            assert_eq!(
                [
                    asked.limit(ChatCompletions),
                    asked.limit(Completions),
                    asked.limit(Responses)
                ],
                [chat, completions, responses],
                "{body}"
            );
        }
    }

    #[test]
    fn a_cap_is_the_lower_of_the_clients_limit_and_itself() {
        let asked = asked(r#"{"max_completion_tokens":30}"#, 1).unwrap();
        assert_eq!(asked.capped(Route::ChatCompletions, 64), 30);
        assert_eq!(asked.capped(Route::ChatCompletions, 24), 24);
        // Completions without max_tokens: vLLM's 16, under any larger cap.
        assert_eq!(asked.capped(Route::Completions, 64), 16);
        assert_eq!(asked.capped(Route::Completions, 8), 8);
        // No limit on Responses: the cap.
        assert_eq!(asked.capped(Route::Responses, 64), 64);
    }

    #[test]
    fn limits_and_choices_that_the_engine_might_read_otherwise_are_refused() {
        for (body, error) in [
            (
                r#"{"max_tokens":1,"max_tokens":9000}"#,
                Invalid::Duplicate("max_tokens"),
            ),
            (
                r#"{"max_output_tokens":null,"max_output_tokens":5}"#,
                Invalid::Duplicate("max_output_tokens"),
            ),
            (r#"{"n":1,"n":8}"#, Invalid::Duplicate("n")),
            (r#"{"max_tokens":"64"}"#, Invalid::NotACount("max_tokens")),
            (
                r#"{"max_completion_tokens":64.0}"#,
                Invalid::NotACount("max_completion_tokens"),
            ),
            (r#"{"max_tokens":1e3}"#, Invalid::NotACount("max_tokens")),
            (r#"{"max_tokens":-1}"#, Invalid::NotACount("max_tokens")),
            (r#"{"max_tokens":true}"#, Invalid::NotACount("max_tokens")),
            (
                r#"{"max_tokens":18446744073709551616}"#,
                Invalid::NotACount("max_tokens"),
            ),
            (
                r#"{"max_output_tokens":[1]}"#,
                Invalid::NotACount("max_output_tokens"),
            ),
            (r#"{"n":0}"#, Invalid::NotAChoice("n")),
            (r#"{"n":"2"}"#, Invalid::NotAChoice("n")),
            (r#"{"best_of":-1}"#, Invalid::NotAChoice("best_of")),
            (
                r#"{"n":8}"#,
                Invalid::TooManyChoices {
                    field: "n",
                    asked: 8,
                    max: 1,
                },
            ),
            (
                r#"{"best_of":2,"use_beam_search":true}"#,
                Invalid::TooManyChoices {
                    field: "best_of",
                    asked: 2,
                    max: 1,
                },
            ),
        ] {
            assert_eq!(asked(body, 1), Err(error.clone()), "{body}");
            assert!(!error.to_string().is_empty());
        }
        // Within max_choices, and nulls, are allowed; other fields are not read.
        for body in [
            r#"{"n":1,"best_of":null}"#,
            r#"{"n":null}"#,
            r#"{"messages":[],"stream":true,"temperature":0.5}"#,
            r#"{"max_tokens":18446744073709551615}"#,
        ] {
            assert!(asked(body, 1).is_ok(), "{body}");
        }
        assert!(asked(r#"{"n":4,"use_beam_search":true}"#, 4).is_ok());
        assert_eq!(
            asked(r#"{"n":5}"#, 4),
            Err(Invalid::TooManyChoices {
                field: "n",
                asked: 5,
                max: 4
            })
        );
    }

    #[test]
    fn each_route_is_written_its_own_fields() {
        let written = |route| {
            let mut json = Vec::new();
            write_limit(&mut json, route, 24);
            String::from_utf8(json).unwrap()
        };
        assert_eq!(
            written(Route::ChatCompletions),
            r#""max_completion_tokens":24,"max_tokens":24,"#
        );
        assert_eq!(written(Route::Completions), r#""max_tokens":24,"#);
        assert_eq!(written(Route::Responses), r#""max_output_tokens":24,"#);
    }

    /// One case of `tests/fixtures/reply-caps.json`: a body of `route` under a class's `cap` and degrade's, and what
    /// the gateway made of it: the body it sends upstream with the reply limit it wrote, or its 400.
    #[derive(serde::Serialize)]
    struct Case {
        route: &'static str,
        cap: Option<u32>,
        degrade: Option<u32>,
        max_choices: u32,
        input: String,
        status: u16,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        /// The reply limit the gateway wrote; none for a body it left as it was.
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        param: Option<&'static str>,
    }

    /// The reply limits of the fixture's bodies: every field absent, null, under the cap and over it, alone and with
    /// the others in either order.
    const LIMITS: [&str; 19] = [
        "",
        r#","max_tokens":4000"#,
        r#","max_tokens":8"#,
        r#","max_tokens":null"#,
        r#","max_completion_tokens":4000"#,
        r#","max_completion_tokens":8"#,
        r#","max_completion_tokens":null"#,
        r#","max_output_tokens":4000"#,
        r#","max_output_tokens":8"#,
        r#","max_output_tokens":null"#,
        r#","max_completion_tokens":8,"max_tokens":4000"#,
        r#","max_completion_tokens":4000,"max_tokens":8"#,
        r#","max_completion_tokens":null,"max_tokens":8"#,
        r#","max_completion_tokens":null,"max_tokens":4000"#,
        r#","max_tokens":null,"max_completion_tokens":8"#,
        r#","max_output_tokens":8,"max_tokens":4000"#,
        r#","max_output_tokens":4000,"max_completion_tokens":8"#,
        r#","max_output_tokens":8,"max_completion_tokens":4000,"max_tokens":4000"#,
        r#","max_completion_tokens":null,"max_tokens":null,"max_output_tokens":null"#,
    ];

    /// Reply limits the gateway refuses: each field twice, the second time over the cap, and values that are not
    /// counts.
    const REFUSED: [&str; 9] = [
        r#","max_tokens":8,"max_tokens":4000"#,
        r#","max_completion_tokens":8,"max_completion_tokens":4000"#,
        r#","max_output_tokens":8,"max_output_tokens":4000"#,
        r#","max_tokens":"4000""#,
        r#","max_completion_tokens":4000.0"#,
        r#","max_output_tokens":4e3"#,
        r#","max_tokens":-1"#,
        r#","max_tokens":18446744073709551616"#,
        r#","max_tokens":[8]"#,
    ];

    /// Choices, with the max_choices of the gateway they are sent to.
    const ASKED_CHOICES: [(&str, u32); 10] = [
        (r#","n":1"#, 1),
        (r#","n":null"#, 1),
        (r#","n":2"#, 1),
        (r#","n":2"#, 2),
        (r#","n":4,"use_beam_search":true"#, 4),
        (r#","n":5,"use_beam_search":true"#, 4),
        (r#","n":0"#, 1),
        (r#","best_of":2"#, 1),
        (r#","n":1,"n":8"#, 1),
        (r#","n":3,"best_of":3"#, 4),
    ];

    /// Every case of the fixture, through the gateway's own rewrite: each route with each arrangement of limits under
    /// a class's cap or none and with degrade's or none, the refused limits under a cap, and the choices.
    fn reply_cap_cases() -> Vec<Case> {
        let metrics = crate::metrics::Metrics::new();
        let routes = [
            (
                Route::ChatCompletions,
                r#""model":"qwen","messages":[{"role":"user","content":"hi"}]"#,
            ),
            (Route::Completions, r#""model":"qwen","prompt":"hi""#),
            (Route::Responses, r#""model":"qwen","input":"hi""#),
        ];
        let mut inputs = Vec::new();
        for (route, content) in routes {
            for limits in LIMITS {
                for (cap, degrade) in [
                    (Some(24), None),
                    (None, None),
                    (Some(24), Some(16)),
                    (None, Some(16)),
                ] {
                    inputs.push((route, format!("{{{content}{limits}}}"), cap, degrade, 1));
                }
            }
            for limits in REFUSED {
                inputs.push((route, format!("{{{content}{limits}}}"), Some(24), None, 1));
            }
            for (choices, max_choices) in ASKED_CHOICES {
                inputs.push((
                    route,
                    format!("{{{content}{choices}}}"),
                    Some(24),
                    None,
                    max_choices,
                ));
            }
        }
        inputs
            .into_iter()
            .map(|(route, input, cap, degrade, max_choices)| {
                let bounds = crate::sanitize::Bounds {
                    route,
                    cap,
                    max_choices,
                };
                let case = |status| Case {
                    route: route.as_str(),
                    cap,
                    degrade,
                    max_choices,
                    input: input.clone(),
                    status,
                    output: None,
                    limit: None,
                    code: None,
                    param: None,
                };
                match crate::sanitize::rewrite_body(
                    input.as_bytes(),
                    "salt",
                    None,
                    bounds,
                    &metrics,
                ) {
                    Ok(body) => {
                        let json = match degrade {
                            Some(degrade) => crate::sanitize::cap_reply(&body.json, route, degrade),
                            None => body.json,
                        };
                        let crate::sanitize::TopLevel(members) =
                            serde_json::from_str(&input).unwrap();
                        let asked =
                            read(members.iter().map(|(k, v)| (k.as_str(), *v)), max_choices)
                                .unwrap();
                        let cap = [cap, degrade].into_iter().flatten().min();
                        Case {
                            output: Some(String::from_utf8(json.to_vec()).unwrap()),
                            limit: cap.map(|cap| asked.capped(route, cap)),
                            ..case(200)
                        }
                    }
                    Err(crate::sanitize::InvalidBody::Bounds(invalid)) => Case {
                        code: Some(invalid.code()),
                        param: Some(invalid.field()),
                        ..case(400)
                    },
                    Err(err) => panic!("{input}: {err}"),
                }
            })
            .collect()
    }

    /// `tests/fixtures/reply-caps.json` is what the gateway makes of each case now: the file that
    /// `python/flatshare_vllm/tests/test_gateway_caps.py` parses with vLLM 0.32.0's own request models.
    /// `FLATSHARE_GATEWAY_WRITE_FIXTURES=1` writes it anew (`make gateway-fixtures`).
    #[test]
    fn the_reply_caps_fixture_is_current() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reply-caps.json");
        let cases: Vec<String> = reply_cap_cases()
            .iter()
            .map(|case| serde_json::to_string(case).unwrap())
            .collect();
        let about = "The gateway's rewrite of reply limits and choices (src/bounds.rs), case by case: a body of a \
                     route, under a class's cap and degrade's (null: none) and max_choices, and the body the gateway \
                     sends upstream with the limit it wrote (status 200), or its 400. Written by `make \
                     gateway-fixtures`, checked by the gateway's tests, and read by \
                     python/flatshare_vllm/tests/test_gateway_caps.py, which parses each body with vLLM 0.32.0's own \
                     request models.";
        let fixture = format!(
            "{{\n  \"about\": {},\n  \"cases\": [\n    {}\n  ]\n}}\n",
            serde_json::to_string(about).unwrap(),
            cases.join(",\n    ")
        );
        if std::env::var_os("FLATSHARE_GATEWAY_WRITE_FIXTURES").is_some() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &fixture).unwrap();
        }
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            on_disk == fixture,
            "{} is stale; write it anew with `make gateway-fixtures`",
            path.display()
        );
        // The fixture holds every kind of case: capped and not, refused for each reason, and the old bypass, a chat
        // body with only max_output_tokens, capped.
        let parsed: serde_json::Value = serde_json::from_str(&fixture).unwrap();
        let cases = parsed["cases"].as_array().unwrap();
        for code in [
            "duplicate_field",
            "invalid_reply_limit",
            "invalid_choices",
            "too_many_choices",
        ] {
            assert!(cases.iter().any(|case| case["code"] == code), "{code}");
        }
        let bypass = cases
            .iter()
            .find(|case| {
                case["route"] == "/v1/chat/completions"
                    && case["cap"] == 24
                    && case["degrade"].is_null()
                    && case["input"]
                        .as_str()
                        .unwrap()
                        .ends_with(r#""max_output_tokens":4000}"#)
            })
            .unwrap();
        assert_eq!(bypass["limit"], 24);
        assert!(
            bypass["output"]
                .as_str()
                .unwrap()
                .starts_with(r#"{"max_completion_tokens":24,"max_tokens":24,"#)
        );
    }
}
