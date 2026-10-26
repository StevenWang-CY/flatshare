# The budget cap's window, swept (2026-10-25)

`python/flatshare_vllm/src/flatshare_vllm/make-room/knee.sh` on Phastform's RTX 5060 Ti, on the frozen box's units and pool at v0.3-191-g987fd09 (`DEPLOYED`), on 2026-10-25 15:27–16:29 UTC: make-room with the PRs' switches (two preemptions a step, the budget cap on, no reserve) at a window of 16, 24, 32, 48 and 64 tokens, each read back from the engine's `/server_info` (`wWINDOW-engine.json`), and for each window two 240 s runs of `tail.py` beside the box's closed-loop agents, with fresh prompts and call replies of 40 tokens timed to their first, 16th and last token (`wWINDOW-RUN.out`). The engine's make-room lines over each window's runs are in `wWINDOW-make-room.log`, the sweep's own log in `knee.log`. `table.py` prints the table below from the outputs.

The medians of each window's two runs, in ms (per run in `table.py`'s first table); no turn failed in any run:

| Window (tokens) | First token p95 | 16th token p95 | 40th token p95 | Agents' prompt tokens/s |
|---:|---:|---:|---:|---:|
| 16 | 448 | 726 | 3,759 | 3,483 |
| 24 | 387 | 648 | 2,642 | 3,443 |
| 32 | 326 | 572 | 1,854 | 3,300 |
| 48 | 347 | 616 | 1,037 | 3,205 |
| 64 | 366 | 612 | 1,018 | 3,246 |

- **The reply's tail falls until the window covers the reply.** Past its window a reply decodes beside agent prefill again, so its 40th token comes 3.8 s after the request at 16 tokens and 1.0 s once the window holds all 40; a window longer than the reply changes nothing more (48 against 64).
- **What the agents pay stops with the reply too.** Their prompt throughput falls from 3,483 to 3,205 tokens a second, 8%, as the window grows to cover the reply, and no further at 64: a hold ends when its reply ends, so a window longer than every reply costs no more than one just long enough.
- **The first token does not depend on the window**, which acts only after it: 326–448 ms at p95 across the sweep.

The window to deploy follows: long enough to cover the reply the voice agent speaks first, its first sentence, at its longest ([PRODUCTION.md](../../../docs/PRODUCTION.md#the-replys-first-tokens)). These replies are fixed at 40 tokens; E2's voice agent's replies ran 23 tokens at the median and 52 at most, and E2 measured 16 and 48.
