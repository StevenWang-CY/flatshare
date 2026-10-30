# Security policy

How to report a vulnerability in Flatshare, and which parts of the repository the report covers.

## Supported versions

Fixes go into the latest release and `main`. At v1.11 that is:

| Version | Supported |
|---|---|
| 1.11 (the latest release) and `main` | Yes |
| 1.10 and earlier, including the releases named Leaseholder | No |

## Reporting a vulnerability

Report it privately, through GitHub's security advisories for this repository: on the repository's **Security** tab,
choose **Report a vulnerability**, or open
[github.com/StevenWang-CY/flatshare/security/advisories/new](https://github.com/StevenWang-CY/flatshare/security/advisories/new).
Do not open a public issue or pull request for it. Say which version or commit you tested, which component, how to
reproduce the problem, and what an attacker gains. The maintainer answers in the advisory's thread, and the fix and its
disclosure are agreed there.

## Scope

In scope:

- **`flatshare-gateway`** ([`crates/flatshare-gateway`](crates/flatshare-gateway/README.md)): API-key authentication,
  the tenant and class a request gets, the request fields and headers it strips because the engines trust them, the
  scoping of session ids to tenants, the per-tenant `cache_salt`, the metrics key, and its listeners (loopback TCP and
  Unix sockets).
- **The deploy scripts and units** ([`deploy/`](deploy)): what they leave listening on a host and with what file
  modes, the SSH tunnel keys' `restrict` entries and forced commands, the network namespaces of the GPU servers and
  the voice plane, the listener audit, and the secrets the install scripts write.

- **The serving engine's build** ([`python/flatshare_vllm`](python/flatshare_vllm/README.md)): the module `build.py`
  writes from make-room's patches, the serve module that the `make-room-serve` variant loads, and
  `deploy/bin/serving-check`, which holds the engine to the serving profile.

Report anything else in the repository the same way. It is measurement code (the other engine variants, the benchmark
harness, E2's admission policies in the gateway, the simulator), and a security issue there is fixed as a bug.
Vulnerabilities in vLLM, SGLang, LiveKit, SMG or another project Flatshare runs belong to that project's own security
process.

These limits are stated in the documents and are not vulnerabilities in this sense:

- The voice path keeps callers' audio and transcripts in memory until the call's end, and its metadata in the
  hosts' journals for as long as journald keeps them, a retention the operator sets
  ([what the voice path keeps](docs/PRODUCTION.md#what-the-voice-path-keeps)).
- The engine's Unix sockets keep the hosts' other users out, not other processes of the deploying user, and the
  voice plane's processes keep that user's uid ([ADR-0005](docs/adr/0005-loopback-services-and-node-sticky-kv.md#unix-sockets-end-to-end-w6-2026-10-25)).
- The `sglang` engine variant listens on 127.0.0.1:8000, and a host without unprivileged user namespaces runs its GPU
  servers in the host's network namespace ([ARCHITECTURE.md §4](docs/ARCHITECTURE.md#33-network-and-security)).
- The measurement drop-in turns on vLLM's development routes, which the serving variant turns off again, so a production
  box runs without them ([the drop-in](deploy/phastform/variants/README.md#the-measurement-drop-in)).
- The gateway does not terminate TLS: every listener binds loopback or a private socket. It bounds the requests in
  flight and the bodies they hold, and gives each class deadlines for its upload, its queue and its response, but it
  does not bound the idle connections clients keep open
  ([what it does not do](crates/flatshare-gateway/README.md#what-it-does-not-do)).

## Fixes in pinned dependencies

A release pins everything it serves ([THIRD-PARTY.md](THIRD-PARTY.md)):
- vLLM 0.32.0, with make-room's patches built into its scheduler;
- the Python environments' locks;
- LiveKit server 1.13.10 and LiveKit Agents 1.8.8;
- Kokoro-FastAPI at its measured commit;
- the models at their revisions.

Their advisories are published by their projects: the GitHub security advisories of vLLM, PyTorch and LiveKit, and
the OSV database for the locked Python packages. A vulnerability in one of them goes to its project. One in how
Flatshare uses or pins it is reported here, as above.

A fix in a pinned dependency ships as a Flatshare release that moves the pin. Like any release, it goes through CI
and the release tool, whose checks hold the box to it ([PRODUCTION.md](docs/PRODUCTION.md#releasing-a-change)):
- **A fix on the pinned line**, such as a vLLM 0.32.x, needs make-room's patches to apply to the new version first.
- **A fix only in a later line** needs make-room's patches rebased onto that line first.
  - `make compat-upstream` says whether they still apply
    ([the engine's upkeep](python/flatshare_vllm/README.md#upkeep-the-changes-cost-and-its-fit-upstream)).
  - The rebased engine must pass the same checks before a release moves to it.

### Advisories the pins carry

Two pins carry advisories that cannot reach a Flatshare host. Each stays for the reason given, and moves when that
reason goes. `python/flatshare_bench/tests/test_security_versions.py` holds each to the files it names.

- **`setuptools==80.10.2`**, in the engine's environment (`deploy/phastform/requirements/engine.txt`), the A100
  kit's (`deploy/a100/requirements/engine-cu129.txt`) and CI's CPU environment for make-room
  (`python/flatshare_vllm/requirements-cpu.txt`).
  - PYSEC-2026-3447 (CVE-2026-59890), fixed in 83.0.0: on macOS, a file name in Unicode's decomposed form can slip
    past a MANIFEST.in exclusion when an sdist is built.
  - GHSA-xmj3-9gf7-h52w, fixed in 84.1.0: on Windows, a wheel installed the legacy way can write an `__init__.py`
    outside its directory.
  - It stays because vLLM 0.32.0 requires `setuptools<81`. The engine's hosts run Linux and install pinned wheels,
    so neither path runs there. The pin moves with the vLLM release that lifts that bound.
- **`diskcache==5.6.3`**, in the SGLang variant's environment (`deploy/phastform/variants/sglang/requirements.txt`).
  - PYSEC-2026-2447 (CVE-2025-69872), with no fixed release: pickle data that an attacker writes into the cache
    directory runs as code when it is read.
  - The variant is a comparator in E2's records and is not deployed, and its cache directory belongs to the deploying
    user.

The bench's lock takes Pillow 13.0.0, which fixes 39 advisories against 12.3.0, over AIPerf 0.13.0's `pillow~=12.3.0`
(the override in `python/flatshare_bench/pyproject.toml`). AIPerf's image paths give the same bytes under it.
