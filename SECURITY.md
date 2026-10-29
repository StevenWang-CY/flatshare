# Security policy

How to report a vulnerability in Flatshare, and which parts of the repository the report covers.

## Supported versions

Fixes go into the latest release and `main`. At v1.7 that is:

| Version | Supported |
|---|---|
| 1.7 (the latest release) and `main` | Yes |
| 1.6 and earlier, including the releases named Leaseholder | No |

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

- The engine's Unix sockets keep the hosts' other users out, not other processes of the deploying user, and the
  voice plane's processes keep that user's uid ([ADR-0005](docs/adr/0005-loopback-services-and-node-sticky-kv.md#unix-sockets-end-to-end-w6-2026-10-25)).
- The `sglang` engine variant listens on 127.0.0.1:8000, and a host without unprivileged user namespaces runs its GPU
  servers in the host's network namespace ([ARCHITECTURE.md §3.3](docs/ARCHITECTURE.md#33-network-and-security)).
- The measurement drop-in turns on vLLM's development routes, which the serving variant turns off again, so a production
  box runs without them ([the drop-in](deploy/phastform/variants/README.md#the-measurement-drop-in)).
- The gateway does not terminate TLS: every listener binds loopback or a private socket. It bounds the requests in
  flight and the bodies they hold, and gives each class deadlines for its upload, its queue and its response, but it
  does not bound the idle connections clients keep open
  ([what it does not do](crates/flatshare-gateway/README.md#what-it-does-not-do)).
