//! The error of the token estimate (`flatshare_core::admission::estimate_tokens`) on an AIPerf replay, against the
//! prompt tokens the engine reported.
//!
//! `cargo run --release -p flatshare-core --example estimate_error -- RUN/aiperf/profile_export.jsonl...`
//!
//! Each record of AIPerf's per-request export has the bytes the request sent (`http_req_data_sent`: its
//! headers and its body), the prompt tokens the engine reported, and the root correlation id of its session
//! tree, which the `flatshare_chat` endpoint sends as `X-Session-ID`. The records are replayed as the
//! gateway sees them: a request is estimated when it starts, from the last report of its session that had
//! ended by then. The error is given over all prompt tokens, and per request for requests of 1,000 tokens or
//! more, beside what bytes / 4 alone would give.

use std::collections::HashMap;

use flatshare_core::admission::{SessionUsage, estimate_tokens};
use serde_json::Value;

/// One request of the replay.
struct Record {
    start_ns: u64,
    end_ns: u64,
    session: String,
    bytes: u64,
    prompt_tokens: u64,
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: estimate_error PROFILE_EXPORT.jsonl...");
        std::process::exit(2);
    }
    for path in paths {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let records: Vec<Record> = text.lines().filter_map(record).collect();
        let (anchored, fallback, with_report) = estimates(&records);
        println!(
            "{path}: {} requests, {with_report} with an earlier report of their session",
            records.len()
        );
        println!("  the estimate:  {}", describe(&records, &anchored));
        println!("  bytes / 4:     {}", describe(&records, &fallback));
    }
}

fn record(line: &str) -> Option<Record> {
    let record: Value = serde_json::from_str(line).ok()?;
    let (meta, metrics) = (&record["metadata"], &record["metrics"]);
    let value = |name: &str| metrics[name]["value"].as_f64();
    let session = meta["root_correlation_id"]
        .as_str()
        .or_else(|| meta["x_correlation_id"].as_str())?;
    Some(Record {
        start_ns: meta["request_start_ns"].as_u64()?,
        end_ns: meta["request_end_ns"].as_u64()?,
        session: session.to_owned(),
        // AIPerf reports kilobytes of 1,024 bytes.
        bytes: (value("http_req_data_sent")? * 1024.0).round() as u64,
        prompt_tokens: value("usage_prompt_tokens")? as u64,
    })
}

/// Each request's estimate from its session's last report, and from its bytes alone, with the number of
/// requests whose session had reported before they started.
fn estimates(records: &[Record]) -> (Vec<u64>, Vec<u64>, usize) {
    // Starts and ends in time order, an end first at the same instant.
    let mut events: Vec<(u64, bool, usize)> = records
        .iter()
        .enumerate()
        .flat_map(|(i, r)| [(r.start_ns, true, i), (r.end_ns, false, i)])
        .collect();
    events.sort_unstable();
    let mut last: HashMap<&str, SessionUsage> = HashMap::new();
    let mut anchored = vec![0; records.len()];
    let mut with_report = 0;
    for (_, start, i) in events {
        let r = &records[i];
        if start {
            let usage = last.get(r.session.as_str()).copied();
            with_report += usize::from(usage.is_some());
            anchored[i] = estimate_tokens(r.bytes, usage).tokens;
        } else {
            let usage = SessionUsage {
                prompt_tokens: r.prompt_tokens,
                body_bytes: r.bytes,
            };
            last.insert(&r.session, usage);
        }
    }
    let fallback = records
        .iter()
        .map(|r| estimate_tokens(r.bytes, None).tokens)
        .collect();
    (anchored, fallback, with_report)
}

/// The estimates' error over all prompt tokens, and its quantiles over the requests of 1,000 tokens or more. An
/// export may have neither: an engine that reported no usage, or a replay of short prompts.
fn describe(records: &[Record], estimates: &[u64]) -> String {
    let reported: u64 = records.iter().map(|r| r.prompt_tokens).sum();
    let off: u64 = records
        .iter()
        .zip(estimates)
        .map(|(r, &e)| e.abs_diff(r.prompt_tokens))
        .sum();
    let mut errors: Vec<f64> = records
        .iter()
        .zip(estimates)
        .filter(|(r, _)| r.prompt_tokens >= 1_000)
        .map(|(r, &e)| e.abs_diff(r.prompt_tokens) as f64 / r.prompt_tokens as f64)
        .collect();
    errors.sort_by(f64::total_cmp);
    let total = if reported == 0 {
        "no prompt tokens reported".to_owned()
    } else {
        format!(
            "off by {:.2}% of all prompt tokens",
            off as f64 / reported as f64 * 100.0
        )
    };
    let Some(last) = errors.len().checked_sub(1) else {
        return format!("{total}; no request of 1,000 tokens or more");
    };
    let within = |q: f64| errors[(last as f64 * q).round() as usize] * 100.0;
    format!(
        "{total}; of {} requests of 1,000 tokens or more, 50% within {:.1}%, 90% within {:.1}%, 99% within \
         {:.1}%, all within {:.1}%",
        errors.len(),
        within(0.5),
        within(0.9),
        within(0.99),
        within(1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(start_ns: u64, end_ns: u64, bytes: u64, prompt_tokens: u64) -> Record {
        Record {
            start_ns,
            end_ns,
            session: "s".to_owned(),
            bytes,
            prompt_tokens,
        }
    }

    #[test]
    fn a_request_is_estimated_from_its_sessions_last_report_that_had_ended() {
        let records = [
            record(0, 10, 8_000, 1_900),
            // Sent while the first was in flight: no report yet, its bytes alone.
            record(5, 20, 8_400, 2_000),
            // After both ended: the second's report and the 400 bytes since, 100 tokens.
            record(30, 40, 8_800, 2_100),
        ];
        let (anchored, fallback, with_report) = estimates(&records);
        assert_eq!(anchored, [2_000, 2_100, 2_100]);
        assert_eq!(fallback, [2_000, 2_100, 2_200]);
        assert_eq!(with_report, 1);
        assert_eq!(
            describe(&records, &anchored),
            "off by 3.33% of all prompt tokens; of 3 requests of 1,000 tokens or more, 50% within 5.0%, 90% \
             within 5.3%, 99% within 5.3%, all within 5.3%"
        );
    }

    #[test]
    fn an_export_without_long_requests_or_usage_is_described_too() {
        // Every prompt under 1,000 tokens: no quantile to give, where the error over all tokens stands (500 tokens
        // for each, 100 off each).
        let short = [record(0, 10, 2_000, 400), record(20, 30, 2_400, 600)];
        let (anchored, _, _) = estimates(&short);
        assert_eq!(anchored, [500, 500]);
        assert_eq!(
            describe(&short, &anchored),
            "off by 20.00% of all prompt tokens; no request of 1,000 tokens or more"
        );
        // No request at all, as from an export whose records all lacked their usage.
        assert_eq!(
            describe(&[], &[]),
            "no prompt tokens reported; no request of 1,000 tokens or more"
        );
    }
}
