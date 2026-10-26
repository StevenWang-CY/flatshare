"""The statistics Flatshare's measurements publish, each defined once.

- :func:`quantiles`, and :func:`summarize` and :func:`percentiles` on it: every percentile is the
  inclusive one, linear between the closest ranks, as numpy's ``percentile`` and flatshare-sim's are, so
  that a reader's tool recomputes the same p95 from the raw samples.
- :func:`v2v_samples`: the voice-to-voice samples of the callers' turn lines, a reply that never
  came censored at the reply timeout.
- :func:`metric_samples`: the samples of a Prometheus text exposition, as vLLM and the gateway
  export them.

It needs only the standard library. The bench and voice packages depend on it; the probes, which
run in the engine's environment (python/flatshare_vllm/src/flatshare_vllm/make-room/), and
flatshare-sim's calibration script load it from the source tree by path.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

PERCENTILES = (10, 50, 90, 95, 99)
"""Reported by :func:`summarize`; the SLOs are set at p95 (docs/SLO.md)."""


def quantiles(samples: Iterable[float], n: int = 100) -> list[float]:
    """The ``n - 1`` cut points that divide the samples into ``n`` groups (by default the 1st to the
    99th percentile), by the inclusive method: the k-th at rank ``k / n * (len - 1)`` of the sorted
    samples, linear between the closest ranks. A single sample is every cut point; none is a
    ValueError."""
    xs = sorted(samples)
    if not xs:
        raise ValueError("no samples")
    if len(xs) == 1:
        return [xs[0]] * (n - 1)
    return statistics.quantiles(xs, n=n, method="inclusive")


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Count, mean and :data:`PERCENTILES` of a sample, to a hundredth; only the count of none."""
    xs = sorted(values)
    if not xs:
        return {"n": 0}
    qs = quantiles(xs)
    cell: dict[str, float | int] = {"n": len(xs), "mean": round(statistics.fmean(xs), 2)}
    return cell | {f"p{p}": round(qs[p - 1], 2) for p in PERCENTILES}


def percentiles(samples: Iterable[float], censored: int = 0) -> dict[str, float | int]:
    """Count, p50, p95 and p99 of voice-to-voice samples in ms, to a tenth of a millisecond, as the
    callers' summary and E2's report give them; ``censored`` of the samples stand at the reply
    timeout (:func:`v2v_samples`). A single sample gives only its p50, itself."""
    xs = sorted(samples)
    count = {"n": len(xs)} | ({"censored": censored} if censored else {})
    if len(xs) < 2:
        return count | ({"p50": xs[0]} if xs else {})
    q = quantiles(xs)
    return count | {"p50": round(q[49], 1), "p95": round(q[94], 1), "p99": round(q[98], 1)}


def v2v_samples(turns: Iterable[Mapping[str, Any]], timeout_ms: float) -> tuple[list[float], int]:
    """The voice-to-voice samples of the callers' turn lines and how many of them are censored: each
    turn's reply (``v2v_ms``) and, after a barge-in, the reply to the interruption
    (``v2v_after_barge_in_ms``), a reply that never came standing at ``timeout_ms``. The caller
    leaves out the turns it does not measure."""
    samples, censored = [], 0
    for turn in turns:
        wanted = ["v2v_ms", *(["v2v_after_barge_in_ms"] if turn.get("barge_in") else [])]
        for key in wanted:
            censored += key not in turn
            samples.append(turn.get(key, timeout_ms))
    return samples, censored


_VALUE = r'[^"\\\n]*(?:\\[\\"n][^"\\\n]*)*'
"""A label value between its quotes: any character but the three the format escapes, which are."""
_LABEL = rf'[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*=[ \t]*"({_VALUE})"[ \t]*'
_LABEL_SYNTAX = rf'[ \t]*[a-zA-Z_][a-zA-Z0-9_]*[ \t]*=[ \t]*"{_VALUE}"[ \t]*'
_SAMPLE = re.compile(
    r"[ \t]*(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    rf"(?:\{{(?P<labels>(?:{_LABEL_SYNTAX}(?:,{_LABEL_SYNTAX})*,?)?)[ \t]*\}})?"
    r"[ \t]+(?P<value>[^ \t#]+)"
    r"(?:[ \t]+(?P<timestamp>[^ \t#]+))?"
    r"(?:[ \t]+#.*)?[ \t]*"
)
"""A sample's line: its name, its labels, its value, then perhaps a timestamp and an exemplar."""
_LABELS = re.compile(_LABEL)
_ESCAPED = re.compile(r"\\(.)")
_UNESCAPED = {"\\": "\\", '"': '"', "n": "\n"}


def metric_samples(exposition: str) -> Iterator[tuple[str, dict[str, str], float]]:
    """Every sample of a Prometheus text exposition (format 0.0.4, or OpenMetrics) as ``(name,
    labels, value)``, in its order. A label value is unescaped (``\\\\``, ``\\"``, ``\\n``) and
    may hold any other character, ``}`` and ``,`` among them; a sample's timestamp and its exemplar
    (``# {...} value``) are read past. Comments, blank lines and lines that are not a sample (a
    value or timestamp that is not a number, an escape the format has not) are skipped."""
    for line in exposition.split("\n"):
        m = _SAMPLE.fullmatch(line.removesuffix("\r"))
        if m is None:
            continue
        try:
            value = float(m["value"])
            if m["timestamp"] is not None:
                float(m["timestamp"])
        except ValueError:
            continue
        labels = {
            name: _ESCAPED.sub(lambda e: _UNESCAPED[e[1]], text) if "\\" in text else text
            for name, text in _LABELS.findall(m["labels"] or "")
        }
        yield m["name"], labels, value
