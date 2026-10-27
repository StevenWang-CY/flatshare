from __future__ import annotations

import math
import random
import statistics

import pytest

from flatshare_stats import metric_samples, percentiles, quantiles, summarize, v2v_samples


def linear(xs: list[float], p: float) -> float:
    """The p-th percentile as numpy's ``percentile`` computes it by default: linear between the
    closest ranks of the sorted samples."""
    v = sorted(xs)
    rank = p / 100 * (len(v) - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (rank - lo) * (v[hi] - v[lo])


def test_the_percentiles_are_linear_between_the_closest_ranks() -> None:
    rng = random.Random(7)
    for size in (2, 3, 47, 50, 73, 101):
        xs = [rng.lognormvariate(7, 1) for _ in range(size)]
        cuts = quantiles(xs)
        assert len(cuts) == 99
        for p in range(1, 100):
            assert cuts[p - 1] == pytest.approx(linear(xs, p), rel=1e-12)
        twentieths = quantiles(xs, 20)
        assert twentieths[18] == pytest.approx(cuts[94], rel=1e-12)  # p95 either way
    # Not the exclusive method, statistics.quantiles' default, which with few samples puts the tail
    # higher, its p99 past the largest of them.
    xs = [float(x) for x in range(1, 21)]
    assert (quantiles(xs)[94], quantiles(xs)[98]) == pytest.approx((19.05, 19.81))
    exclusive = statistics.quantiles(xs, n=100)
    assert (exclusive[94], exclusive[98]) == pytest.approx((19.95, 20.79))


def test_one_sample_is_every_percentile_and_none_is_refused() -> None:
    assert quantiles([7.0]) == [7.0] * 99
    assert quantiles([7.0], 4) == [7.0] * 3
    with pytest.raises(ValueError, match="no samples"):
        quantiles([])


def test_summarize_reports_the_count_the_mean_and_the_percentiles() -> None:
    cell = summarize(float(x) for x in range(1, 101))
    assert cell["n"] == 100 and cell["p50"] == 50.5 and cell["p90"] == 90.1
    assert cell["p95"] == 95.05  # the SLO percentile
    assert (cell["mean"], cell["p10"], cell["p99"]) == (50.5, 10.9, 99.01)
    single = {"n": 1, "mean": 7.0, "p10": 7.0, "p50": 7.0, "p90": 7.0, "p95": 7.0, "p99": 7.0}
    assert summarize([7.0]) == single
    assert summarize([]) == {"n": 0}
    two = {"n": 2, "mean": 1.5, "p10": 1.1, "p50": 1.5, "p90": 1.9, "p95": 1.95, "p99": 1.99}
    assert summarize([1.004, 2.0]) == two  # to a hundredth


def test_the_v2v_percentiles_count_the_censored_samples() -> None:
    samples = [float(x) for x in range(1, 101)]
    assert percentiles(samples) == {"n": 100, "p50": 50.5, "p95": 95.0, "p99": 99.0}
    assert percentiles([*samples[:98], 30_000.0, 30_000.0], censored=2) == {
        "n": 100,
        "censored": 2,
        "p50": 50.5,
        "p95": 95.0,
        "p99": 30_000.0,
    }
    assert percentiles([1600.0]) == {"n": 1, "p50": 1600.0}
    assert percentiles([]) == {"n": 0}


def test_the_v2v_samples_take_each_reply_and_censor_the_ones_that_never_came() -> None:
    turns = [
        {"turn": 0, "v2v_ms": 1600.0},
        {"turn": 1, "barge_in": True, "v2v_ms": 1700.0, "v2v_after_barge_in_ms": 1500.0},
        {"turn": 2, "error": "no reply"},  # the turn's reply never came
        {"turn": 3, "barge_in": True, "v2v_ms": 1650.0},  # nor the interruption's
    ]
    assert v2v_samples(turns, 30_000.0) == ([1600.0, 1700.0, 1500.0, 30_000.0, 1650.0, 30_000.0], 2)
    assert v2v_samples([], 30_000.0) == ([], 0)


def test_metric_samples_reads_every_sample_and_skips_the_rest() -> None:
    exposition = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:cache_config_info{block_size="16",num_gpu_blocks="2275"} 1.0
flatshare_gateway_first_byte_seconds_bucket{class="realtime",le="+Inf"} 20
flatshare_gateway_up 1
not a sample
"""
    assert list(metric_samples(exposition)) == [
        ("vllm:num_requests_running", {"engine": "0", "model_name": "m"}, 3.0),
        ("vllm:cache_config_info", {"block_size": "16", "num_gpu_blocks": "2275"}, 1.0),
        ("flatshare_gateway_first_byte_seconds_bucket", {"class": "realtime", "le": "+Inf"}, 20.0),
        ("flatshare_gateway_up", {}, 1.0),
    ]


def test_metric_samples_reads_the_named_metrics_alone() -> None:
    exposition = """\
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
vllm:num_requests_waiting{engine="0"} 2.0
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 2.0
vllm:kv_cache_usage_perc{engine="0"} 0.5
  vllm:kv_cache_usage_perc{engine="1"} 0.25
vllm:num_requests_waiting one
vllm:num_requests_running{engine="0"} 3.0
"""
    # A name that only begins a wanted one is another metric's, and blanks before a sample are the format's.
    assert list(
        metric_samples(exposition, ("vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"))
    ) == [
        ("vllm:num_requests_waiting", {"engine": "0"}, 2.0),
        ("vllm:kv_cache_usage_perc", {"engine": "0"}, 0.5),
        ("vllm:kv_cache_usage_perc", {"engine": "1"}, 0.25),
    ]
    assert list(metric_samples(exposition, ())) == []
    with pytest.raises(TypeError, match="not one"):
        next(metric_samples(exposition, "vllm:num_requests_waiting"))
    # The samples of any names are those of the whole exposition's with those names, in its order.
    every = list(metric_samples(exposition))
    for names in ({"vllm:num_requests_running"}, {"vllm:num_requests_waiting_by_reason", "absent"}):
        assert list(metric_samples(exposition, names)) == [s for s in every if s[0] in names]


def test_metric_samples_follows_the_text_format() -> None:
    # A line with a timestamp or an exemplar was not read, and a label value with a brace or an escaped quote was
    # cut at it: the sample was lost, or its labels were wrong.
    exposition = (
        # A timestamp (the text format's milliseconds, OpenMetrics' seconds) and an exemplar are read past.
        'http_requests_total{method="post",code="200"} 1027 1395066363000\n'
        'flatshare_gateway_requests_total{class="realtime"} 7 # {trace_id="KOMLtNiG"} 1 1700000000.5\n'
        'flatshare_gateway_requests_created{class="realtime"} 1.7e9 1700000000.25\n'
        # A label value holds any character; a backslash, a double quote and a line feed come escaped.
        'msg{text="a } b, c=\\"d\\"",path="C:\\\\DIR\\\\",lines="1\\n2"} 1\n'
        # Blanks around a label's parts, a trailing comma, an empty set, the special values, a CRLF line end.
        'spaced{ a = "1" , b="2", } 2\n'
        "empty{} 3\n"
        "nan NaN\n"
        "up +Inf\n"
        "down -Inf\n"
        "crlf 4\r\n"
        # Not samples: a value or a timestamp that is not a number, an escape the format has not, a label value
        # without quotes, labels with no comma between them, an unclosed set, a number after the timestamp.
        "bad_value one\n"
        "bad_timestamp 1 yesterday\n"
        'bad_escape{a="\\t"} 1\n'
        "bad_quotes{a=1} 1\n"
        'bad_comma{a="1" b="2"} 1\n'
        'bad_brace{a="1" 1\n'
        "bad_tail 1 2 3\n"
        "# EOF\n"
    )
    samples = list(metric_samples(exposition))
    assert [(name, labels) for name, labels, _ in samples] == [
        ("http_requests_total", {"method": "post", "code": "200"}),
        ("flatshare_gateway_requests_total", {"class": "realtime"}),
        ("flatshare_gateway_requests_created", {"class": "realtime"}),
        ("msg", {"text": 'a } b, c="d"', "path": "C:\\DIR\\", "lines": "1\n2"}),
        ("spaced", {"a": "1", "b": "2"}),
        ("empty", {}),
        ("nan", {}),
        ("up", {}),
        ("down", {}),
        ("crlf", {}),
    ]
    values = [value for _, _, value in samples]
    assert values[:6] == [1027.0, 7.0, 1.7e9, 1.0, 2.0, 3.0]
    assert math.isnan(values[6]) and values[7:] == [math.inf, -math.inf, 4.0]
    # Read by its name alone, each metric gives the samples the whole exposition gives it, and a line that is not a
    # sample is skipped as there.
    for name in {name for name, _, _ in samples} | {"bad_value", "bad_escape", "bad_tail"}:
        named = [(n, labels) for n, labels, _ in metric_samples(exposition, (name,))]
        assert named == [(n, labels) for n, labels, _ in samples if n == name], name
