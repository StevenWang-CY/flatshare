"""ab.py: the transport A/B from its records in ab/ (python ab.py, from this directory): flatshare-bench voice-turn's
stages through W5's TCP forwards and through the final build's sockets, p50 / p90 in ms over 30 turns and the p50s'
difference, then each flatshare-bench tiers cell whose median moved by more than 10 ms between the two (the rest
within). Prints Markdown."""

import json
from pathlib import Path

AB = Path(__file__).resolve().parent / "ab"
STAGES = {
    "stt_ms": "STT",
    "llm_ttft_ms": "LLM first token",
    "llm_first_sentence_ms": "LLM first sentence",
    "tts_first_audio_ms": "TTS first audio",
    "turn_ms": "**Turn**",
    "llm_total_ms": "LLM whole reply",
}


def load(name: str) -> dict:
    return json.loads((AB / name / "summary.json").read_text())


def signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "\N{MINUS SIGN}")


tcp, uds = load("tcp-voice-turn"), load("uds-voice-turn")
print("| Stage | TCP forwards (W5's tunnel) | Sockets (flatshare-tunnel) | Sockets − TCP, p50 |")
print("|---|---:|---:|---:|")
for key, label in STAGES.items():
    a, b = tcp[key], uds[key]
    print(f"| {label} | {a['p50']:.1f} / {a['p90']:.1f} | {b['p50']:.1f} / {b['p90']:.1f} | {signed(b['p50'] - a['p50'])} |")
print()
a, b = load("tcp-tiers")["p50_ms"], load("uds-tiers")["p50_ms"]
cells = [(tier, n) for tier in a for n in a[tier]]
moved = [(tier, n) for tier, n in cells if abs(b[tier][n] - a[tier][n]) > 10]
print(
    f"flatshare-bench tiers: {len(cells) - len(moved)} of {len(cells)} cells' medians within 10 ms of each other",
    end="",
)
print("; the others, TCP / sockets:" if moved else ".")
for tier, n in moved:
    print(f"- {tier} at {int(n):,} tokens: {a[tier][n]:,.1f} / {b[tier][n]:,.1f} ms")
