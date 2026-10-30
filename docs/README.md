# Documentation

A map of Flatshare's documentation for a reader new to the project, in the order to read it. Each line names a document and the question it answers. The [repository's README](../README.md) gives the result on one page.

## Start here

- [REPORT.md](REPORT.md): Why does a voice turn wait seconds beside coding agents on one GPU, what do the two scheduler rules recover, at what cost to the agents, and what carries to other hardware?
- [ARCHITECTURE.md](ARCHITECTURE.md): How is the deployed system built, from the change inside vLLM's scheduler to the gateway, the voice worker, two cards with a warm spare and releases that roll back, and what was measured and dropped?
- [GLOSSARY.md](GLOSSARY.md): What does a term or a record's label mean (the frozen box, arm 9, W0 to W8, the project's former names)?

## Deploy and operate

- [PRODUCTION.md](PRODUCTION.md): How do I deploy the change beside a voice agent, is my deployment in the regime where it pays, what should I watch, and how does it fail?
- [HARDWARE.md](HARDWARE.md): What does the deployment need from another GPU or a single host, and how is a node rendered there?
- [SLO.md](SLO.md): Which service classes share the GPU, which indicators are measured for each, and what is each campaign's SLO?
- [PROTOCOL.md](PROTOCOL.md): What does the gateway accept from callers and send to the engine: sessions, headers and lifecycle events?
- [ops/RUNBOOK.md](../ops/RUNBOOK.md): How are the hosts deployed, checked, operated and recovered, and how are the failure drills run?

## The evidence

- [BENCHMARKS.md](BENCHMARKS.md): Where does each number come from, question by question, with its runs, ranges and records (the A100 run's among them), and what did the campaigns before W5 measure?
- [EXPERIMENTS.md](EXPERIMENTS.md): How was each experiment (E0 to E3) designed and run, and which campaign measured what, on which code (the campaign map)?
- [benchmarks/quality/GATE.md](../benchmarks/quality/GATE.md): Can the served model do both jobs of a useful pair: answer a caller's questions on private documents, and enrich new documents into searchable records in the background?
- [benchmarks/quality/RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md): At the same service to the calls, which arrangement of two cards does the most background work, and is sharing them worth its complexity?
- [PRIOR-ART.md](PRIOR-ART.md): How does Flatshare differ from the engines, routers and papers that co-serve latency classes, and what does it not claim?
- [benchmarks/results](../benchmarks/results): Where are the committed records behind each number, one directory per campaign or probe, each with its README?
- [REVISIONS.md](REVISIONS.md): Which commit of this history holds each revision the records name, since the history was rewritten on 2026-10-27?

## Decisions

- [ADR-0001](adr/0001-control-plane-in-front-of-stock-vllm.md): Why was a control plane in front of stock vLLM built first, and what did E2 change about it?
- [ADR-0002](adr/0002-one-lifecycle-event-schema.md): Why do voice and agents share one lifecycle-event schema?
- [ADR-0003](adr/0003-realtime-is-never-shed.md): Why does no admission policy refuse a realtime request, and where is its room made?
- [ADR-0004](adr/0004-native-cpu-offload-tier.md): Why is the host tier vLLM's own CPU offload, and why was lease-aware eviction never switched on?
- [ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md): How are the services kept off the hosts' networks, and what stays open on a shared host?
- [ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md): Why does the change carry the realtime tier, with no admission policy in front?
- [ADR-0007](adr/0007-the-gateway-is-the-serving-control-plane.md): Why is the gateway the serving control plane, and where do E2's admission policies stay?
- [ADR-0008](adr/0008-two-cards-consolidation-with-a-warm-spare.md): How do the two cards serve, and how does a call fail over between them?
- [ADR-0009](adr/0009-immutable-releases-and-the-exact-serving-profile.md): How is a release made, and how is a box held to the configuration E2 measured?

## Components

- [crates/flatshare-gateway](../crates/flatshare-gateway/README.md): What does the gateway do and not do, how is it configured, and what does it add to a request?
- [crates/flatshare-core](../crates/flatshare-core/README.md): What do the cost, step and KV models and the admission policies compute?
- [crates/flatshare-proto](../crates/flatshare-proto/README.md): Which wire types and JSON Schemas do the components share?
- [crates/flatshare-sim](../crates/flatshare-sim/README.md): How does the simulator model vLLM's KV cache and step loop, how was it calibrated, and what does it place on other GPUs?
- [python/flatshare_vllm](../python/flatshare_vllm/README.md): How is the change built on vLLM 0.32.0, held equal to the pull requests, tested, and checked for cost and fit?
- [python/flatshare_bench](../python/flatshare_bench/README.md): How are the traces scaled and replayed, what do the probes measure, and what does an E2 point record?
- [python/flatshare_voice](../python/flatshare_voice/README.md): How does the voice agent run, which lifecycle events does it post, and how do the callers measure voice-to-voice latency?
- [python/flatshare_stats](../python/flatshare_stats/README.md): How is each published statistic defined?
- [benchmarks/e2](../benchmarks/e2/README.md): Which campaigns and arm files make up E2, and how are they run, reported and turned into the demo video?
- [benchmarks/a100/rehearsal](../benchmarks/a100/rehearsal/README.md): How was the A100 kit rehearsed end to end on the box before the paid run?
- [deploy/phastform/variants](../deploy/phastform/variants/README.md): Which engine variants and flags do E2's arms switch, and how is each checked?

## The project

- [CHANGELOG.md](../CHANGELOG.md): What changed in each release, and what must a deployment change when it upgrades?
- [CONTRIBUTING.md](../CONTRIBUTING.md): How do I build and test the repository, and what does a change need before review?
- [SECURITY.md](../SECURITY.md): How do I report a vulnerability, and which parts of the repository does a report cover?
- [THIRD-PARTY.md](../THIRD-PARTY.md): What does the serving path run that Flatshare does not write, and under which licences?
- [brand/README.md](brand/README.md): How are the logo, the mark and the social preview drawn, and how may they be used?
