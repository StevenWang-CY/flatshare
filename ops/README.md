# Operations

What an operator uses once a deployment serves, for whoever runs the hosts. [The runbook](RUNBOOK.md) is the entry: it
deploys, checks, operates and recovers them. [PRODUCTION.md](../docs/PRODUCTION.md) says what to watch and how the
system fails, and [`deploy/`](../deploy/README.md) holds what each host runs.

| Path | What it is |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | The procedures: deploy a release, check a host, switch the engine's variant, add a second GPU node, recover from each failure |
| [`drills/`](drills) | The failure drills, run from a workstation: an engine kill, a gateway reload, restart and kill, a node failover, and a timed cold start |
| [`prometheus/`](prometheus) | The alert rules, their tests (`make check-rules`) and an example scrape configuration |
| [`grafana/`](grafana) | One dashboard of the gateway and the engine |
| [`metrics/`](metrics) | The metric families vLLM exports, by name and type. A test fails when a rule or a panel names a metric that is neither there nor in [the gateway's](../crates/flatshare-gateway/metrics.txt) |
