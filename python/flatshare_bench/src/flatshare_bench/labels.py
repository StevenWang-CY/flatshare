"""The label table: the name and note of every configuration a figure draws, in docs/GLOSSARY.md's terms (LABELS),
and the configuration each E2 point runs (POINTS). The report's charts and E2's figures both read it, so it sits
below both and imports nothing of theirs."""

from __future__ import annotations

LABELS = {
    # The references.
    "voice": ("Voice alone", "no agents"),
    "stock": ("Stock vLLM", "priority scheduling"),
    "tuned": ("Tuned vLLM", "the baseline"),
    # Stock vLLM's other settings, and what an operator can deploy without the change (W4's discriminating scenario).
    "steps-1024": ("1,024-token steps", "no long-prefill threshold"),
    "watermark": ("A KV watermark", "10%"),
    "queued": ("A queued-token limit", "8,192 tokens"),
    "reservation": ("A token reservation", "2,048 tokens"),
    "cap-2": ("A static cap", "2 agent requests in flight"),
    "cap-8": ("A static cap", "8 agent requests in flight"),
    "pr-54530": ("vLLM with #54530", "at most 12 sequences"),
    "sglang": ("SGLang 0.5.22", "priority with preemption"),
    "hicache": ("SGLang with HiCache", "4x, write-through"),
    "flow": ("llm-d's flow control", "emulated"),
    "smg": ("SMG 1.13.0", "as shipped"),
    "resubmit": ("Preempt-resubmit", "before the first byte"),
    "lifecycle": ("The lifecycle policy alone", "on tuned vLLM"),
    "lifecycle-reserve": ("The lifecycle policy", "over make-room"),
    # The change and its parts.
    "kv-preemption": ("The KV preemption", "#63350"),
    "change": ("The change", "16-token window"),
    "switches-off": ("The change, its switches off", ""),
    "deploy": ("The configuration to deploy", "48-token window"),
}
"""Every configuration a figure names, by its key: its name, in docs/GLOSSARY.md's terms, and the note a row gives
under or beside it. Each figure takes its configurations' labels from here, so that a configuration reads the same in
every figure. Its name alone tells it from every other configuration, so a legend or a line's label gives the name
only; a figure whose rows differ in a setting the figure is about (the step cap's chart, the A100's card) gives that
setting as the note instead. The change is the two pull requests at the budget cap's default 16-token window; the
configuration to deploy is their switches with a 48-token window (docs/PRODUCTION.md), which W5 ran at the frozen
box's 512-token steps and the deployment runs at 256."""
POINTS = {
    "a1-voice": "voice",
    "a1-voice-c4": "voice",
    "a1-voice-c6": "voice",
    "disc-frozen-s24": "tuned",
    "std-frozen-s16": "tuned",
    "a100-tuned-s24": "tuned",
    "disc-a4-b1024-t0-s24": "steps-1024",
    "disc-a5-watermark10-s24": "watermark",
    "disc-a5-queued8192-s24": "queued",
    "disc-a6-reserve2048-s24": "reservation",
    "disc-a6-cap2-s24": "cap-2",
    "disc-a6-cap8-s24": "cap-8",
    "disc-a7a-pr54530-s24": "pr-54530",
    "disc-a7b-sglang-s24": "sglang",
    "disc-a7b-sglang-hicache-s24": "hicache",
    "disc-a8a-flow-s24": "flow",
    "disc-a8b-smg-s24": "smg",
    "disc-a8c-resubmit-s24": "resubmit",
    "disc-a9-gateway-only-s24": "lifecycle",
    "disc-a9-s24": "lifecycle-reserve",
    "disc-pr-a-s24": "kv-preemption",
    "disc-pr-ab-s24": "change",
    "disc-pr-ab48-s24": "deploy",
    "disc-pr-ab48-c4-s24": "deploy",
    "disc-pr-ab48-c6-s24": "deploy",
    "std-pr-ab48-s16": "deploy",
    "sweep-disc-pr-ab48-b256-t0-s24": "deploy",
    "a100-pr-ab48-2048-s24": "deploy",
}
"""The configuration each E2 point a figure draws runs, by the point's name less W5's prefix (W5 re-ran W4's arm
files): its key in LABELS."""


def label(point: str) -> tuple[str, str]:
    """The label of the configuration that ``point`` (an E2 point, or a key of LABELS) runs: its name and its note."""
    key = POINTS.get(point.removeprefix("w5-"), point)
    if key not in LABELS:
        raise ValueError(
            f"no label for {point}: add its configuration to flatshare_bench.labels' POINTS"
        )
    return LABELS[key]


def in_prose(name: str) -> str:
    """A label's name inside a sentence: its first word in lower case where it is an ordinary word ("Tuned", "The",
    "A"), and as it stands where it is a name ("SGLang", "vLLM", "SMG")."""
    first = name.split(" ", 1)[0]
    ordinary = first[0].isupper() and (len(first) == 1 or first[1:].replace("-", "").islower())
    return name[0].lower() + name[1:] if ordinary else name
