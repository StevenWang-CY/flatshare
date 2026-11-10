# Documentation

A map of Flatshare's documents for a reader new to the project, in reading order, each with the question it answers.
The [repository's README](../README.md) gives the result on one page.

## Start here

- [REPORT.md](REPORT.md): Why does a voice turn wait seconds beside coding agents on one GPU, what do the two
  scheduler rules and the engine's additions recover, and at what cost to the agents?
- [ARCHITECTURE.md](ARCHITECTURE.md): How is the system built, from the change in vLLM's scheduler to the gateway, the
  voice worker, two cards and releases?
- [GLOSSARY.md](GLOSSARY.md): What does a term or a record's label mean?
- [PRIOR-ART.md](PRIOR-ART.md): How does Flatshare differ from the engines, routers and papers that co-serve latency
  classes?

## Deploy and operate

- [PRODUCTION.md](PRODUCTION.md): How do I deploy the change beside a voice agent, is my deployment in the regime
  where it pays, what should I watch, and how does it fail?
- [HARDWARE.md](HARDWARE.md): What does the deployment need from another GPU or a single host?
- [SLO.md](SLO.md): Which service classes share the GPU, and what is each one's objective?
- [PROTOCOL.md](PROTOCOL.md): What does the gateway accept from callers, and what does it send to the engine?
- [ops/RUNBOOK.md](../ops/RUNBOOK.md): How are the hosts deployed, checked, operated and recovered?
- [deploy/container](../deploy/container/README.md): How does the engine run from vLLM's own container image?

## The evidence

- [BENCHMARKS.md](BENCHMARKS.md): Where does each number come from, with its runs, ranges and records?
- [EXPERIMENTS.md](EXPERIMENTS.md): How was each experiment designed and run, and which campaign measured what?
- [benchmarks/quality/GATE.md](../benchmarks/quality/GATE.md): Can the served model do a useful pair of background
  jobs?
- [benchmarks/quality/RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md): Which arrangement of two cards does
  the most background work at the same service to the calls?
- [benchmarks/fp8](../benchmarks/fp8/README.md): What crashed the engine with an FP8 KV cache beside the CPU tier?
- [benchmarks/results](../benchmarks/results): Which committed record holds each number?
- [REVISIONS.md](REVISIONS.md): Which commit holds each revision the records name?

## Decisions

- [ADR-0001](adr/0001-control-plane-in-front-of-stock-vllm.md): Why was a control plane in front of stock vLLM built
  first, and what did E2 change?
- [ADR-0002](adr/0002-one-lifecycle-event-schema.md): Why do voice and agents share one lifecycle-event schema?
- [ADR-0003](adr/0003-realtime-is-never-shed.md): Why does no admission policy refuse a realtime request?
- [ADR-0004](adr/0004-native-cpu-offload-tier.md): Why is the host tier vLLM's own CPU offload?
- [ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md): How are the services kept off the hosts' networks?
- [ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md): Why does the engine change carry the realtime
  tier?
- [ADR-0007](adr/0007-the-gateway-is-the-serving-control-plane.md): Why is the gateway the serving control plane?
- [ADR-0008](adr/0008-two-cards-consolidation-with-a-warm-spare.md): How do two cards serve, and how does a call fail
  over between them?
- [ADR-0009](adr/0009-immutable-releases-and-the-exact-serving-profile.md): How is a release made, and how is a box
  held to the measured configuration?

## Components

- [crates/flatshare-gateway](../crates/flatshare-gateway/README.md): What does the gateway do, and how is it
  configured?
- [crates/flatshare-core](../crates/flatshare-core/README.md): What do the cost, step and KV models and the admission
  policies compute?
- [crates/flatshare-proto](../crates/flatshare-proto/README.md): Which wire types do the components share?
- [crates/flatshare-sim](../crates/flatshare-sim/README.md): How does the simulator model vLLM's KV cache, and what
  does it place on other GPUs?
- [python/flatshare_vllm](../python/flatshare_vllm/README.md): How is the change built on vLLM, tested, and checked
  for cost and fit?
- [python/flatshare_bench](../python/flatshare_bench/README.md): How are the traces replayed, and what do the probes
  measure?
- [python/flatshare_voice](../python/flatshare_voice/README.md): How does the voice agent run, and how do the callers
  measure voice-to-voice latency?
- [python/flatshare_stats](../python/flatshare_stats/README.md): How is each published statistic defined?
- [benchmarks/e2](../benchmarks/e2/README.md): Which campaigns make up E2, and how are they run and reported?
- [benchmarks/a100/rehearsal](../benchmarks/a100/rehearsal/README.md): How was the A100 kit rehearsed before the paid
  run?
- [deploy/phastform/variants](../deploy/phastform/variants/README.md): Which engine variants do E2's arms switch?

## The project

- [CHANGELOG.md](../CHANGELOG.md): What changed in each release, and what must a deployment change to upgrade?
- [CONTRIBUTING.md](../CONTRIBUTING.md): How do I build and test the repository, and what does a change need?
- [SECURITY.md](../SECURITY.md): How do I report a vulnerability?
- [THIRD-PARTY.md](../THIRD-PARTY.md): What does the serving path run that Flatshare does not write, under which
  licences?
- [brand/README.md](brand/README.md): How are the logo and the social preview drawn, and how may they be used?
