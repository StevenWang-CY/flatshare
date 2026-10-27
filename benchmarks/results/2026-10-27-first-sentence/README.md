# How long the voice agent's first sentences are (2026-10-27)

make-room's prefill hold keeps lower-priority prefill out of a realtime request's first `budget_cap_tokens` tokens, the
window, so that the reply's first sentence, which the voice agent hands to speech synthesis, decodes on steps of its
own. The window has to cover that sentence. The E2 records keep each reply's token count but not its text, so
[`benchmarks/replies/first_sentence.py`](../../replies/first_sentence.py) asked the engine for the agent's replies
again and counted each reply's tokens up to the one that completes its first sentence.

`replies.jsonl` holds the 320 replies, one a line: 120 to the caller corpus's 40 turn lines on their own (three each),
and 200 in eight conversations of 25 turns that keep their history, as a call's chat context does (lines drawn with
seed 27). Each line gives the reply's text, its tokens (`completion_tokens`), the tokens it took to complete its first
sentence (`first_sentence_tokens`) and whether it had a sentence end before its last token (`has_sentence_end`; a reply
without one is spoken whole, so its first sentence is all of its tokens).

The first sentence is LiveKit Agents' (`flatshare_bench.voice_turn.first_sentence`): the text up to the first sentence
end that leaves at least 20 characters, which a streamed reply shows once the character after it arrives. The prompt
is the voice worker's instructions (`flatshare_voice.worker.INSTRUCTIONS`) and the sampling the engine's default, the
model's generation config. `python/flatshare_bench/tests/test_first_sentence.py` keeps the script's prompt and rule
equal to the worker's and the bench's, and checks the numbers below against `replies.jsonl`.

Measured on Gray's RTX 5060 Ti (driver 580.178.04) with vLLM 0.32.0 and Qwen3-4B-Instruct-2507-FP8 at revision
8591804019c8b22094c3b5b4454e0edc05dffc98, the frozen box's engine at v1.5:

| Replies | First sentence, tokens: median | p95 | Most | Within 16 tokens | Within 48 tokens | Whole reply: median | Most |
|---|---:|---:|---:|---:|---:|---:|---:|
| Each line on its own (120) | 14 | 26 | 31 | 79 | 120 | 26 | 48 |
| In conversations (200) | 16 | 30 | 39 | 101 | 200 | 26 | 43 |
| All (320) | 15 | 29 | 39 | 180 (56%) | 320 (100%) | 26 | 48 |

So the 48-token window covers every first sentence, with 9 tokens to spare at the longest, and the default 16-token
window a little over half of them: the replies whose first sentence runs past 16 tokens decode their later tokens
beside agent prefill, which is what E2 measured at T = 16.

To measure again, with the engine on its socket:

```sh
python benchmarks/replies/first_sentence.py --engine unix:$XDG_RUNTIME_DIR/flatshare/engine.sock \
  --out benchmarks/results/2026-10-27-first-sentence
```
