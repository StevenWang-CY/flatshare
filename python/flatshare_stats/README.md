# flatshare-stats

The statistics Flatshare's measurements publish, each defined once, in the standard library only:

| Function | What it gives |
|---|---|
| `quantiles(samples, n=100)` | The cut points by the inclusive method, linear between the closest ranks, as numpy's `percentile` and flatshare-sim's: every percentile the repository publishes |
| `summarize(values)` | Count, mean, p10, p50, p90, p95 and p99, to a hundredth: the bench's probes and E2's records |
| `percentiles(samples, censored=0)` | Count, p50, p95 and p99 of voice-to-voice samples, to a tenth of a millisecond, with the count of censored samples: the callers' summary and E2's report |
| `v2v_samples(turns, timeout_ms)` | The voice-to-voice samples of the callers' turn lines: each reply and the reply to an interruption, one that never came censored at the reply timeout |
| `replies(turns, quiet_ms=1000)` | The replies the callers heard to their end, from the same lines: how many ended by the agent's state, their audio's length, those with a pause, one over 500 ms and one of `quiet_ms` or more, the longest pause's percentiles and the concealed share: the callers' summary and the soak's |
| `metric_samples(exposition, names=None)` | Every sample of a Prometheus text exposition, as `(name, labels, value)`, or with `names` only those metrics' samples, parsing no other line |

[flatshare-bench](../flatshare_bench/README.md) and [flatshare-voice](../flatshare_voice/README.md) depend on it. The make-room probes, `probe.py` and `tail.py`, run in the engine's environment on the GPU node, from the deployed source tree, and flatshare-sim's `scripts/calibration.py` runs with no project: they load the module from this directory by path.

`tests/test_stats.py` checks the percentiles against linear interpolation between ranks over random samples (and that they are not the exclusive method's), a single sample and none, the summaries' rounding and censored counts, the V2V samples of turns with barge-ins and replies that never came, the replies' summary of turns with and without pauses and receiver counts, the samples an exposition holds, and those of the metrics a reader names. `make test-python` runs them.
