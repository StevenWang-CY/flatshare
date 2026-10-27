"""KV block lifecycle histograms from vLLM's ``/metrics``, compared across a run (``flatshare-bench kv-metrics``).

With ``--kv-cache-metrics``, vLLM 0.32 samples GPU KV blocks and exports:
- ``vllm:kv_block_reuse_gap_seconds``: time between consecutive accesses of a block;
- ``vllm:kv_block_idle_before_evict_seconds``: how long an evicted block sat unused before eviction;
- ``vllm:kv_block_lifetime_seconds``: allocation to eviction.

The comparison is a cheap first signal of retention headroom. If evicted blocks were idle for less
time than blocks typically wait between reuses, LRU is evicting KV that would have been reused.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from flatshare_bench.transport import client
from flatshare_stats import metric_samples

REUSE_GAP = "vllm:kv_block_reuse_gap_seconds"
IDLE_BEFORE_EVICT = "vllm:kv_block_idle_before_evict_seconds"
LIFETIME = "vllm:kv_block_lifetime_seconds"
HISTOGRAMS = (REUSE_GAP, IDLE_BEFORE_EVICT, LIFETIME)


def fetch(address: str, timeout: float = 10.0) -> str:
    """The ``/metrics`` exposition of the server at ``address`` (:mod:`flatshare_bench.transport`): an engine's,
    or the gateway's on its socket."""
    with client(address, timeout=timeout) as http:
        response = http.get("/metrics")
        response.raise_for_status()
        return response.text


Labels = tuple[tuple[str, str], ...]
"""A sample's labels as sorted ``(name, value)`` pairs."""


def samples(exposition: str, name: str) -> dict[Labels, float]:
    """The samples of metric ``name``, by label set; empty when the exposition has none."""
    return {
        tuple(sorted(labels.items())): value
        for _, labels, value in metric_samples(exposition, (name,))
    }


def metric_series(exposition: str) -> dict[str, dict[Labels, float]]:
    """Every metric's samples, by name and then label set, in one pass over the exposition: for each
    name, what :func:`samples` gives for it."""
    by_name: dict[str, dict[Labels, float]] = {}
    for sample, labels, value in metric_samples(exposition):
        by_name.setdefault(sample, {})[tuple(sorted(labels.items()))] = value
    return by_name


@dataclass(frozen=True)
class Histogram:
    """Cumulative bucket counts, ordered by upper bound; the last bound is +Inf."""

    bounds: tuple[float, ...]
    counts: tuple[float, ...]

    @classmethod
    def parse(cls, text: str, name: str, match: Mapping[str, str] | None = None) -> Histogram:
        """Read one histogram from an exposition, summing its label sets (those that carry every
        label in ``match``)."""
        totals: dict[float, float] = {}
        wanted = (match or {}).items()
        for _, labels, value in metric_samples(text, (f"{name}_bucket",)):
            if "le" in labels and labels.items() >= wanted:
                bound = float(labels["le"])  # "+Inf" parses as math.inf
                totals[bound] = totals.get(bound, 0.0) + value
        bounds = tuple(sorted(totals))
        return cls(bounds, tuple(totals[b] for b in bounds))

    def __sub__(self, earlier: Histogram) -> Histogram:
        before = dict(zip(earlier.bounds, earlier.counts, strict=True))
        return Histogram(
            self.bounds,
            tuple(c - before.get(b, 0.0) for b, c in zip(self.bounds, self.counts, strict=True)),
        )

    @property
    def total(self) -> float:
        return self.counts[-1] if self.counts else 0.0

    def quantile(self, q: float) -> float | None:
        """Linear interpolation inside the bucket that crosses quantile ``q``."""
        if self.total <= 0:
            return None
        target = q * self.total
        lower, below = 0.0, 0.0
        for bound, count in zip(self.bounds, self.counts, strict=True):
            if count >= target:
                if math.isinf(bound):
                    return lower  # beyond the largest finite bucket: report its bound
                return lower + (bound - lower) * (target - below) / max(count - below, 1e-12)
            lower, below = bound, count
        return lower

    def share_at_most(self, x: float) -> float | None:
        """The share of observations at or below ``x``, interpolated inside its bucket."""
        if self.total <= 0:
            return None
        lower, below = 0.0, 0.0
        for bound, count in zip(self.bounds, self.counts, strict=True):
            if x <= bound:
                if math.isinf(bound):
                    return below / self.total
                share = below + (count - below) * (x - lower) / max(bound - lower, 1e-12)
                return share / self.total
            lower, below = bound, count
        return 1.0


def compare(before: str, after: str) -> str:
    """A short report of the three histograms over the interval between two expositions."""
    hists = {
        name: Histogram.parse(after, name) - Histogram.parse(before, name) for name in HISTOGRAMS
    }
    lines = []
    for name, hist in hists.items():
        qs = [(q, hist.quantile(q)) for q in (0.1, 0.5, 0.9)]
        rendered = " ".join(f"p{round(q * 100)}={v:.3g}s" for q, v in qs if v is not None)
        lines.append(f"{name}: n={hist.total:.0f} {rendered}".rstrip())
    reuse, idle = hists[REUSE_GAP], hists[IDLE_BEFORE_EVICT]
    for q in (0.5, 0.9):
        gap = reuse.quantile(q)
        share = idle.share_at_most(gap) if gap is not None else None
        if share is not None:
            lines.append(
                f"evicted blocks idle no longer than the reuse-gap p{round(q * 100)} "
                f"({gap:.3g}s): {share:.1%}"
            )
    return "\n".join(lines)
