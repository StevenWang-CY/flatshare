# ADR-0009: Immutable releases, and the serving profile that a release holds the box to

- **Status:** accepted, 2026-10-27. v1.6 was the first release that both hosts took this way, on 2026-10-28.
- **Deciders:** Chuyue Wang

## Context

The external audit of 2026-10-27 found that what a box would deploy was not what E2 measured, and that a deploy could not be undone:
- **The defaults were not the measured configuration.**
  - The make-room variant's drop-in reserved 2,048 tokens and set no window, so it ran 16 tokens, where the configuration to deploy runs no reserve and 48.
  - The voice worker admitted every call (`load_fnc` returned 0), and only the benchmark's callers bounded the calls.
- **A deploy wrote into the live tree.** `deploy.sh` copied the source over `~/flatshare/src`, wrote `DEPLOYED` and ran `install.sh`. A deploy left no manifest, and nothing could switch back: two hosts at one version could hold different trees.

## Decision

- **A serving profile names the configuration to serve** (`deploy/profiles/rtx-5060-ti-serve.toml`):
  - the frozen box's engine running the configuration to deploy (two preemptions a step, the budget cap at 48 tokens, no reserve);
  - the step cap the sweep chose (256 tokens a step, no long-prefill threshold: [the sweep](../../benchmarks/results/2026-10-27-sweep/README.md), confirmed in [the rounds](../../benchmarks/results/2026-10-28-rounds/README.md));
  - the module's hash, the calls the worker admits, and real callers' audio.

  `profile.py` renders it into `deploy/phastform/serving.json`, every setting the engine must run with. A test holds the profile to the sweep's measured point.
- **The box is held to it.**
  - The serving variant's engine writes the configuration it runs at its start, and `deploy/bin/serving-check` fails the start on any setting that differs, naming it.
  - The voice worker's serve mode admits the profile's calls and no more. Its bench mode is E2's.
  - A release runs the same check after its switch.
- **A release is immutable** ([PRODUCTION.md](../PRODUCTION.md#releasing-a-change)). `deploy/deploy.sh HOST` makes `~/flatshare/releases/ID` with a manifest of what it holds: the files' hashes, the locks, the environments by key, the gateway's binary and its build, the models' and tokenizer's revisions, the patches and module, and the profile. Then it:
  - verifies the release before it switches;
  - drains what the switch would cut;
  - switches the `current` link and restarts only the units whose inputs changed;
  - switches back by itself when a check after the switch fails;
  - rolls back to the previous release offline.

## Consequences

- **Equal manifests mean equal deployments**, and a release's checks are where the serving configuration is enforced.
- **A change to serving.json restarts the units that read it**: both engines and the voice worker.
- **Deploys run in attended windows.** A release drains the voice worker's calls and the engine's requests before the restarts it needs (about 100 s for an engine).
- **What the release does not cover** is host state outside the repository. Examples are the tunnels' targets and the tenants' keys, which stay in `~/.config/flatshare`. The gateway's `[[nodes]]` are the same, held to the profile's envelope at each release (ADR-0008).
