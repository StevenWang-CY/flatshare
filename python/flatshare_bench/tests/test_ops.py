"""ops/'s dashboards, alert and recording rules, their promtool tests and scrape configuration: every metric an
expression or a test's input names is one the code exports, from the gateway's checked-in expositions
(crates/flatshare-gateway/metrics.txt, and metrics-nodes.txt for [[nodes]]) and the metric sets of vLLM 0.32.0 and the
voice worker (ops/metrics), and the
rules and scrape jobs are what the docs say. make check-rules runs the promtool tests themselves."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
OPS = REPO / "ops"
FUNCTIONS = {
    "abs", "absent", "and", "avg", "avg_over_time", "bool", "by", "clamp_max", "clamp_min", "count", "delta",
    "group_left", "group_right", "histogram_quantile", "ignoring", "increase", "irate", "max", "max_over_time", "min",
    "offset", "on", "or", "quantile", "rate", "scalar", "sum", "time", "topk", "unless", "vector", "without",
}  # fmt: skip
"""PromQL's functions, aggregations and keywords, which the expressions here use."""
BUILT_IN = {"up"}
"""The series Prometheus makes for each scrape."""


def samples(families: Path) -> set[str]:
    """The sample names of a metric set, ``name type`` per line: a counter's end in _total, a histogram's are _bucket,
    _sum and _count."""
    names = set()
    for line in families.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, kind = line.split()
        if kind == "counter":
            names.add(name.removesuffix("_total") + "_total")
        elif kind == "histogram":
            names |= {f"{name}_bucket", f"{name}_sum", f"{name}_count"}
        else:
            names.add(name)
    return names


def exported() -> set[str]:
    crate = REPO / "crates" / "flatshare-gateway"
    gateway = [
        line
        for exposition in ("metrics.txt", "metrics-nodes.txt")
        for line in (crate / exposition).read_text().splitlines()
    ]
    names = {
        re.match(r"[^{\s]+", line).group(0) for line in gateway if line and not line.startswith("#")
    }
    for metric_set in (OPS / "metrics").glob("*.txt"):
        names |= samples(metric_set)
    return names


def metric_names(expr: str) -> set[str]:
    """The metrics a PromQL expression selects: what is left without its strings, label matchers, ranges, grouping
    labels, numbers and keywords."""
    expr = re.sub(r'"(?:[^"\\]|\\.)*"', '""', expr)
    expr = re.sub(r"\{[^}]*\}", "", expr)
    expr = re.sub(r"\[[^\]]*\]", "", expr)
    expr = re.sub(r"\b(by|without|on|ignoring)\s*\([^)]*\)", "", expr)
    return set(re.findall(r"[A-Za-z_:][A-Za-z0-9_:]*", expr)) - FUNCTIONS


def rules() -> list[dict]:
    groups = yaml.safe_load((OPS / "prometheus" / "rules.yml").read_text())["groups"]
    return [rule for group in groups for rule in group["rules"]]


def dashboard_exprs() -> list[str]:
    exprs = []
    for path in (OPS / "grafana").glob("*.json"):
        for panel in json.loads(path.read_text())["panels"]:
            exprs += [target["expr"] for target in panel.get("targets", [])]
    return exprs


def test_every_metric_the_rules_and_dashboards_name_is_exported() -> None:
    known = exported() | BUILT_IN
    recorded = {rule["record"] for rule in rules() if "record" in rule}
    exprs = [rule["expr"] for rule in rules()] + dashboard_exprs()
    assert len(exprs) > 30
    unknown = {name: expr for expr in exprs for name in metric_names(expr) - known - recorded}
    assert unknown == {}, unknown
    # The check reads names: one the code does not export fails it.
    assert metric_names('rate(flatshare_gateway_first_bytes_seconds_bucket{le="0.25"}[5m])') - known


def test_the_rules_name_their_records_and_alerts_as_the_docs_do() -> None:
    for rule in rules():
        if "record" in rule:
            assert re.fullmatch(r"flatshare:[a-z0-9_]+:[a-z0-9_]+", rule["record"]), rule["record"]
        else:
            assert rule["labels"]["severity"] in {"page", "ticket"}, rule["alert"]
            assert rule["annotations"]["summary"], rule["alert"]
    alerts = {rule["alert"] for rule in rules() if "alert" in rule}
    # docs/SLO.md's two objectives, each with a fast and a slow burn.
    for objective in ("FlatshareRealtimeFirstByte", "FlatshareOverhead"):
        assert {f"{objective}BudgetBurn", f"{objective}BudgetSlowBurn"} <= alerts
    sli = {rule["record"]: rule["expr"] for rule in rules() if "record" in rule}
    for window in ("5m", "30m", "1h", "6h", "3d"):
        # The realtime objective reads the failure-inclusive service level, good over valid by class, not the first
        # byte's histogram, which has no sample of a request that failed or hung before its first byte.
        miss = sli[f"flatshare:sli_miss:ratio_rate{window}"]
        for counter in ("flatshare_gateway_sli_good_total", "flatshare_gateway_sli_valid_total"):
            assert f"{counter}[{window}]" in miss
        assert "by (class)" in miss and "first_byte" not in miss
        assert 'le="0.002"' in sli[f"flatshare:overhead_slow:ratio_rate{window}"]
    burns = {rule["alert"]: rule["expr"] for rule in rules() if "alert" in rule}
    for alert in (
        "FlatshareRealtimeFirstByteBudgetBurn",
        "FlatshareRealtimeFirstByteBudgetSlowBurn",
    ):
        assert "flatshare:sli_miss:" in burns[alert] and 'class="realtime"' in burns[alert]
    # The series the objectives read exist in the gateway's exposition.
    gateway = (REPO / "crates" / "flatshare-gateway" / "metrics.txt").read_text()
    for counter in ("flatshare_gateway_sli_good_total", "flatshare_gateway_sli_valid_total"):
        assert f'{counter}{{class="realtime"}}' in gateway
    assert 'flatshare_gateway_overhead_seconds_bucket{le="0.002",' in gateway


def test_every_alert_opens_its_row_in_the_runbook() -> None:
    # An alert's runbook_url is the runbook on main at the alert's own anchor: an id on the row of its Failures table
    # that names the alert. So a new alert needs its row, and every anchor there names an alert.
    runbook = (OPS / "RUNBOOK.md").read_text()
    failures = runbook.split("\n## Failures\n", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in failures.splitlines():
        for anchor in re.findall(r'<a id="([^"]+)"></a>', line):
            rows[anchor] = line
    base = "https://github.com/StevenWang-CY/flatshare/blob/main/ops/RUNBOOK.md#"
    alerts = [rule for rule in rules() if "alert" in rule]
    for rule in alerts:
        alert = rule["alert"]
        assert rule["annotations"]["runbook_url"] == base + alert.lower(), alert
        assert f"`{alert}`" in rows.get(alert.lower(), ""), f"{alert}: no row names it"
    assert set(rows) == {rule["alert"].lower() for rule in alerts}


def test_the_rule_tests_feed_exported_series_and_test_the_objectives_alerts() -> None:
    tests = yaml.safe_load((OPS / "prometheus" / "rules.test.yml").read_text())
    assert tests["rule_files"] == ["rules.yml"]
    known = exported() | BUILT_IN
    series = [s["series"] for test in tests["tests"] for s in test.get("input_series", [])]
    unknown = {s for s in series if metric_names(s) - known}
    assert series and unknown == set(), unknown
    alerts = {rule["alert"] for rule in rules() if "alert" in rule}
    tested = {t["alertname"] for test in tests["tests"] for t in test.get("alert_rule_test", [])}
    assert tested <= alerts, tested - alerts
    assert {
        "FlatshareRealtimeFirstByteBudgetBurn",
        "FlatshareRealtimeFirstByteBudgetSlowBurn",
        "FlatshareEngineDegraded",
        "FlatshareEngineWarming",
        "FlatshareNodeUnhealthy",
        "FlatshareNodeCovering",
        "FlatshareCallsRefused",
        "FlatshareRequestsRefusedForCapacity",
        "FlatshareDeadlinesMissed",
    } <= tested


def test_prometheus_reads_the_gateway_and_the_engine_with_the_metrics_key() -> None:
    config = yaml.safe_load((OPS / "prometheus" / "prometheus.example.yml").read_text())
    assert config["rule_files"] == ["<home>/flatshare/src/ops/prometheus/rules.yml"]
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert set(jobs) == {"flatshare-gateway", "flatshare-engine"}
    for job, path in (("flatshare-gateway", "/metrics"), ("flatshare-engine", "/metrics/engine")):
        assert jobs[job]["metrics_path"] == path
        assert jobs[job]["authorization"] == {
            "credentials_file": "<home>/.config/flatshare/metrics-key"
        }
        assert jobs[job]["static_configs"][0]["targets"] == ["127.0.0.1:18080"]
