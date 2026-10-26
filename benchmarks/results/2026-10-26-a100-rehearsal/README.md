# The release's A100 kit, rehearsed on the box (2026-10-26)

The records of [`benchmarks/a100/rehearsal`](../../a100/rehearsal/README.md)'s run of the release's kit on Phastform,
the box's RTX 5060 Ti host, standing in for the A100's: every step from `deploy/a100/push.sh` to `make reproduce-a100`
in one pass, under flatshare-svc in a home of its own beside the box's deployment, which the rehearsal left as it found
it (`cleanup`: 253,381 lines of units and files alike). The home was new, so every cache the kit fills started empty:
the venvs, the compile and JIT caches, the gateway's build. The models were read from the box's hub cache, offline,
where the A100's first stage downloads them. The node was rendered from the A100's profile (`a100-40gb`) with
Phastform's facts, so these are the box's numbers through the A100's kit: they check its mechanics, as its own preflight
says of a host with 16 CPUs ("this host checks the mechanics, not the numbers").

[The rehearsal of 2026-10-25](../2026-10-25-a100-rehearsal/README.md) found the two defects of the kit that this kit
has fixed. A run of the release's candidate earlier the same night, from a fresh home too, passed every step but one:
`reproduce`'s copy of this workstation's corpus to a peacock that had none failed, because GNU rsync makes a
destination's last directory alone. That was the rehearsal's own script, fixed in f974b2a with a test that fails
without it under GNU rsync. flatshare-svc's two fixes after that run (a start's state written before its runner runs,
each write under a lock; a start whose command is missing fails, as systemd's does) are in the kit this run checked; the
candidate's run measured the same first start within 2.3 s.

Home paths are cut to `~` (the box's login home; the rehearsal's own is `~/a100-rehearsal`), the workstation's to
`$REHEARSAL_WORK` and `$REHEARSAL_REPO`, the account name to `USER`, the hosts' domain names to their SSH aliases, and
LiveKit's key is redacted as its secret is.

| File | What |
|---|---|
| `steps.log` | `rehearse.sh`'s log of the run: each step's start and end, UTC |
| `node/` | the node's facts and resolved values (`host.json`, `node.json`, `staged.json`), `bootstrap.sh`'s log, the preflight (`preflight.json`), the warm starts (`warm.json`) and FlashInfer's JIT check |
| `blocks.log`, `run.log` | `run.sh`'s blocks 0, 1 and 4 |
| `block0/` | `smoke.sh` with the listener audit, and a 2-turn call |
| `block1/` | E0 on the frozen engine (`frozen-e0`, 3 repetitions) and the probes: the KV case and the prefill case three times each, the calls' tail twice |
| `e2/a100-r1/` | round 1's first point, the PRs' 48-token configuration, as `flatshare-bench e2 export` keeps it, with the campaign's log and its report |
| `a100.md` | `make reproduce-a100` over these records |

## The steps

The minutes of each step, from `steps.log`:

| Step | Minutes | |
|---|---:|---|
| `check`, `init`, `data` | 1.3 | Kokoro copied from the box's checkout; weka-x32 and the callers' clips copied from cyan, the long traces made |
| `push` | 1.7 | the source and 2.7 GB of data as files |
| `stage` | 9.1 | uv, Python 3.12 and the venvs, rustup and `cargo build -p flatshare-gateway`, LiveKit's server, FlashInfer's sampler for sm120 (179.0 s), the control plane up |
| `gpu` | 1.4 | the card's facts, vLLM's parse of both configurations, the preflight |
| `warm` | 28.8 | the four compile configurations from empty caches: frozen's three servers 1,232.1 s to ready (the STT 502.9, the TTS 34.8, the engine 692.4), then 176.9, 152.5 and 152.6 s |
| `block0` | 4.1 | `smoke.sh`, the listener audit (the three GPU servers in network namespaces of their own), a 2-turn call |
| `block1` | 16.1 | E0, three KV and three prefill probes, two 240 s tails |
| `block4` | 13.3 | the first point, then the deadline's stop before the second |
| `pull`, `reproduce`, `cleanup` | 0.6 | 4.3 MB pulled; the report on peacock; the box compared and found as it was |

The GPU was in use from `gpu` to `block4`, 64 minutes; the steps took 76. Most of `warm` is the first start's
one-time work, in the servers' own logs: of the STT's 448.8 s of init, torch.compile took 13.5 s and FlashInfer's
autotuning 125; of the engine's 617.5 s, torch.compile 82.4 s, FlashInfer's autotuning 95 and the CUDA graph capture
178. Each later configuration, its engine alone restarted, compiled its own graphs over the warm kernel caches in
153–177 s.

## The box's numbers through the kit

- **The probes (block 1), frozen engine:** the KV case's priority-0 first token after 4,853–4,929 ms (3 runs; Gray's,
  the second 5060 Ti, 4,923–5,077 ms); the prefill case's token gap 61–64 ms (3 runs; Gray's 69 ms); the calls'
  first-token p95 beside 12 agents 7,643 and 7,631 ms over two 240 s tails of 44 turns each, none failed.
- **E2 (block 4):** the PRs' 48-token configuration on W5's round-1 sample of the agents, with the control plane on
  the GPU node's own 16 CPUs: V2V p50 1,950 ms, p95 2,333 ms, none of 50 turns failed, the agents' goodput 3,425
  tokens/s. The box's two-host deployment measured this configuration at a p95 of 2,132–2,397 ms over W5's fifteen
  runs.
- **block 0's call:** two turns, V2V 1,911 and 1,351 ms.
