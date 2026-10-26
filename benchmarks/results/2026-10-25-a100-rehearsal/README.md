# The A100 kit, rehearsed on the box (2026-10-25)

The records of [`benchmarks/a100/rehearsal`](../../a100/rehearsal/README.md)'s run on Phastform, the box's RTX 5060 Ti
host, standing in for the A100's: every step of the kit from `deploy/a100/push.sh` to `make reproduce-a100`, under
flatshare-svc in a home of its own beside the box's deployment, which the rehearsal left as it found it (`cleanup`:
253,192 lines of units and files alike). The node was rendered from the A100's profile (`a100-40gb`) with Phastform's
facts, so these are the box's numbers through the A100's kit: they check its mechanics, as its own preflight says of a
host with 16 CPUs ("this host checks the mechanics, not the numbers").

Home paths are cut to `~` (the box's login home; the rehearsal's own is `~/a100-rehearsal`), the workstation's to
`$REHEARSAL_WORK` and `$REHEARSAL_REPO`, the account name to `USER`, and LiveKit's key is redacted as its secret is.

| File | What |
|---|---|
| `steps.log` | `rehearse.sh`'s log of its three runs: each step's start and end, UTC |
| `node/` | the node's facts and resolved values (`host.json`, `node.json`, `staged.json`), `bootstrap.sh`'s log, the preflight (`preflight.json`), the warm starts (`warm.json`) and FlashInfer's JIT check |
| `blocks.log`, `run.log` | `run.sh`'s blocks 0, 1 and 4 |
| `block0/` | `smoke.sh` with the listener audit, and a 2-turn call |
| `block1/` | E0 on the frozen engine (`frozen-e0`, 3 repetitions) and the probes: the KV case and the prefill case three times each, the calls' tail twice |
| `e2/a100-r1/` | round 1's first point, the PRs' 48-token configuration, as `flatshare-bench e2 export` keeps it, with the campaign's log and its report |
| `a100.md` | `make reproduce-a100` over these records |

## What it found

Two runs stopped at defects of the kit, each fixed with a test that failed first before the third run went on:

- **`bootstrap.sh gpu` failed** at `profile.py check-vllm`: run as a script, deploy/profile.py stood in for the
  standard library's `profile` module when vLLM's import reached cProfile ("module 'profile' has no attribute
  'run'"). Fixed in be9c536.
- **Under flatshare-svc, `warm` started no engine.** vLLM's engine core blanks its /proc/PID/environ when it sets its
  title, where flatshare-svc's FLATSHARE_SVC_UNIT stood, so gpu-guard took the STT's engine core for a foreign process
  and refused the TTS's and the engine's starts (e85b13a); the listener audit read the same environment (4cb5d5f).

## The steps

The minutes of each step's run that passed, from `steps.log`:

| Step | Minutes | |
|---|---:|---|
| `check`, `init`, `data` | 0.4 | the callers' clips copied from cyan |
| `push` | 1.9 | the source and 2.7 GB of data as files; later pushes, the source alone, took 3 s |
| `stage` | 9.1 | uv, Python 3.12 and the venvs (the engine's 5.0 minutes), rustup and `cargo build -p flatshare-gateway`, LiveKit's server, FlashInfer's sampler for sm120 (177.9 s), the control plane up |
| `gpu` | 1.4 | the card's facts, vLLM's parse of both configurations, the preflight |
| `warm` | 21.7 | the four compile configurations: frozen 797.7 s to ready, then 178.8, 152.3 and 152.4 s; the engine's caches were empty, but the STT's were warm from the second run, which had started it ([the release's rehearsal](../2026-10-26-a100-rehearsal/README.md), from a fresh home: 1,232.1 s) |
| `block0` | 4.1 | `smoke.sh`, the listener audit (the three GPU servers in network namespaces of their own), a 2-turn call |
| `block1` | 16.1 | E0, three KV and three prefill probes, two 240 s tails |
| `block4` | 14.2 | the first point, then the deadline's stop before the second |
| `pull`, `reproduce`, `cleanup` | 0.8 | 4.5 MB pulled; the report on peacock; the box compared and found as it was |

The GPU was in use from `gpu` to `block4`, 57 minutes; the steps that passed took 70.

## The box's numbers through the kit

- **The probes (block 1), frozen engine:** the KV case's priority-0 first token after 4,849–4,922 ms (3 runs; Gray's,
  the second 5060 Ti, 4,923–5,077 ms); the prefill case's token gap 62 ms (2 runs; Gray's 69 ms); the calls'
  first-token p95 beside 12 agents 7,631–7,635 ms over two 240 s tails of 44 turns each, none failed.
- **E2 (block 4):** the PRs' 48-token configuration on W5's round-1 sample of the agents, with the control plane on
  the GPU node's own 16 CPUs: V2V p50 1,936 ms, p95 2,109 ms, none of 50 turns failed, the agents' goodput 3,531
  tokens/s. The box's two-host deployment measured this configuration at a p95 of 2,132–2,397 ms over W5's fifteen
  runs.
- **block 0's call:** two turns, V2V 1,854 and 1,354 ms.
