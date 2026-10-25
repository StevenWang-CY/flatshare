# W6 transport records (2026-10-25)

The records behind [BENCHMARKS](../../../docs/BENCHMARKS.md#the-control-plane-lh-gateway-in-front-of-the-engine)'
numbers for the final transport: every GPU server on a Unix socket on Phastform, reached from cyan through
`lh-tunnel@phastform`, one SSH session per connection (v0.3-191-g987fd09, the frozen engine with the measurement
drop-in), measured from cyan with lh-bench's venv.

| File | What it holds |
|---|---|
| `llm_paths.py`, `llm-paths.json` | The paired probe, 12:19Z: the voice agent's LLM request (greedy, up to 60 tokens) in 40 rounds after 2 warm-up ones, one request on each path, the tunnel's engine socket and the gateway's voice socket, their order alternating from round to round; each request's first token, first sentence (the voice worker's cut, 20 characters at least), end and gaps between chunks, the paired medians and whether both paths generated the same text (`python llm_paths.py llm-paths.json 40`, on cores 24-31) |
| `conn-cost.py`, `conn-cost.json` | What a connection through the tunnel costs, 12:36Z: `GET /health` on each of the tunnel's three sockets, 50 on one kept connection, then 50 on connections of their own (`Connection: close`), 50 ms apart (`python conn-cost.py conn-cost.json`, on cores 16-31) |
