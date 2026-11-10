# flatshare-proto

The wire types every Flatshare component shares, for a developer who changes what crosses a process boundary. Each is
defined here once ([ADR-0002](../../docs/adr/0002-one-lifecycle-event-schema.md)), and
[PROTOCOL.md](../../docs/PROTOCOL.md) gives the contract.

| Module | What it defines |
|---|---|
| `event` | Lifecycle events (`POST /v1/events`): one schema for agents and voice ([PROTOCOL.md](../../docs/PROTOCOL.md#lifecycle-events)) |
| `headers` | The headers and body fields the control plane reads, sets and strips ([PROTOCOL.md](../../docs/PROTOCOL.md#headers)) |
| `class`, `session` | The SLO class and the session id the others carry |

## JSON Schemas

`schema/` holds the schemas generated from these types, the reference for the Python components, and `make schema`
regenerates them. The voice worker's `tests/test_events.py` reads the lifecycle event's, and the benchmark's and the
voice worker's `tests/test_protocol.py` the SLO class's.

## Tests

`cargo test -p flatshare-proto` runs `tests/wire.rs`, which pins the wire format with the examples in
`testdata/events.jsonl`, and `tests/schema.rs`, which fails when the schemas fall behind the code.
