# Rust crates

One Cargo workspace of four crates, for a developer who builds or changes them. `make lint-rust` and `make test-rust`
check them, and none is published to crates.io.

| Crate | What it is | Depends on |
|---|---|---|
| [`flatshare-gateway`](flatshare-gateway/README.md) | The serving control plane in front of the engines: it authenticates, bounds, places and relays every request | `flatshare-core`, `flatshare-proto` |
| [`flatshare-proto`](flatshare-proto/README.md) | The wire types the components share: service classes, headers, sessions and lifecycle events, with their JSON Schemas | — |
| [`flatshare-core`](flatshare-core/README.md) | The cost, step-time and KV-geometry models, and the admission policies E2's harness ran. Pure code with no I/O | `flatshare-proto` |
| [`flatshare-sim`](flatshare-sim/README.md) | The simulator of vLLM's KV cache under the agents' traces, and its forecasts for other GPUs | `flatshare-core` |

[ARCHITECTURE.md](../docs/ARCHITECTURE.md#at-a-glance) places each crate in the system.
