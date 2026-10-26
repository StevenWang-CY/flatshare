# W6 transport records (2026-10-25)

The records behind [BENCHMARKS](../../../docs/BENCHMARKS.md#the-control-plane-flatshare-gateway-in-front-of-the-engine)'
numbers for the final transport: every GPU server on a Unix socket on Phastform, reached from cyan through
`flatshare-tunnel@phastform`, one SSH session per connection (the frozen engine with the measurement drop-in), measured
from cyan with flatshare-bench's venv; and the transport A/B against W5's, one SSH connection's TCP forwards to the
servers' ports. Home paths are written `~`.

| File | What it holds |
|---|---|
| `llm_paths.py`, `llm-paths.json` | The paired probe, at v0.3-191-g987fd09 at 12:19Z: the voice agent's LLM request (greedy, up to 60 tokens) in 40 rounds after 2 warm-up ones, one request on each path, the tunnel's engine socket and the gateway's voice socket, their order alternating from round to round; each request's first token, first sentence (the voice worker's cut, 20 characters at least), end and gaps between chunks, the paired medians and whether both paths generated the same text (`python llm_paths.py llm-paths.json 40`, on cores 24-31) |
| `conn-cost.py`, `conn-cost.json` | What a connection through the tunnel costs, on push B (v0.3-274-gd6d6814) at 17:10Z: `GET /health` on each of the tunnel's three sockets, 50 on one kept connection, then 50 on connections of their own (`Connection: close`), 50 ms apart, each mode on a client of its own whose first request is not measured, p95 the nearest rank (`python conn-cost.py conn-cost.json`, on cores 16-31) |
| `conn-cost-tcp.py`, `conn-cost-tcp.json` | The same through W5's forwards, for the A/B's TCP side, on push B at 17:43Z: one SSH connection from cyan with local TCP forwards to the three sockets, as W5's tunnel unit held them to the servers' ports, so that each new connection is a channel on it (`python conn-cost-tcp.py conn-cost-tcp.json engine=PORT,stt=PORT,tts=PORT`, on cores 16-31) |
| `ab/tcp-voice-turn/`, `ab/tcp-tiers/` | The A/B's TCP side, 09:57Z, at v0.3-90-gd43b46d: `flatshare-bench voice-turn --reps 30` over the two LibriSpeech clips and `flatshare-bench tiers --reps 5`, through W5's tunnel (`run.json`, `summary.json`, and each turn or sample) |
| `ab/uds-voice-turn/`, `ab/uds-tiers/` | Its socket side, on push B at 17:01Z: the same commands, clips, reps and cores through `flatshare-tunnel@phastform`, with `voice-turn` reading each TTS reply to its end (04d6d1a), so that its TTS requests keep their connection as the voice worker's do |
| `ab.py` | Prints BENCHMARKS' A/B table from `ab/`: each stage's p50 / p90 on both sides and the p50s' difference, then the `tiers` cells whose medians differ by more than 10 ms (`python ab.py`) |
