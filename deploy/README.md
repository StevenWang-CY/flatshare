# Deployment

What each host of a Flatshare deployment runs, for an operator who deploys it or reads how the measured box is built.
[PRODUCTION.md](../docs/PRODUCTION.md) says what to deploy and why, [the runbook](../ops/RUNBOOK.md) gives the
commands, and [HARDWARE.md](../docs/HARDWARE.md) covers another GPU or a single host.

## Hosts

The reference deployment has a control host and one or two GPU nodes. Each host has a directory named after it, and
`deploy.sh HOST` releases the directory that HOST's first label names, so `phastform.example.org` gets `phastform/`.

| Directory | Host | Holds |
|---|---|---|
| [`phastform/`](phastform) | The GPU node: vLLM, Whisper and Kokoro on one RTX 5060 Ti, each a `systemd --user` unit on a Unix socket | `engine.yaml`, `stt.yaml`, the units, `serving.json`, and [the engine's variants](phastform/variants/README.md) |
| [`gray/`](gray) | The second GPU node, deployed as Phastform is | Its drop-ins over Phastform's units |
| [`cyan/`](cyan) | The control host: the gateway, LiveKit, the voice worker and a tunnel to each GPU node | Their units, [`gateway.example.toml`](cyan/gateway.example.toml), and the worker's and gateway's variants |
| [`a100/`](a100) | One rented host that is both GPU node and control host, as the A100 run used | Its bootstrap, the push of the kit and the pull of its records |

A GPU node's engine and speech configurations, its units and its `make-room` drop-ins are rendered, not written by
hand. `profile.py render` makes them from a hardware profile and [`templates/`](templates), and a test holds
`phastform/`'s to its profile byte for byte ([the rules](../docs/HARDWARE.md#the-rules)). The rest of `phastform/`,
its scripts and the other variants, is kept by hand.

## Shared parts

| Path | What it is |
|---|---|
| [`profiles/`](profiles) | The hardware profiles a node renders from. `rtx-5060-ti.toml` is the measured box, and `rtx-5060-ti-serve.toml` is the serving profile, the configuration to deploy |
| [`templates/`](templates) | The engine and speech configurations and units that a profile fills in |
| [`profile.py`](profile.py) | Detects a host's facts, renders a node from a profile, and checks a rendering against its repository copy |
| [`deploy.sh`](deploy.sh) | Releases this commit to a host as an immutable release, and rolls one back |
| [`build-gateway.sh`](build-gateway.sh) | Builds the gateway's binary on a Linux host, for a release to the control host |
| [`authorize-tunnel.sh`](authorize-tunnel.sh), [`authorize-guard.sh`](authorize-guard.sh) | Give the control host's keys their one forced command on a GPU node |
| [`container/`](container/README.md) | The serving engine as an image built on vLLM's own |
| [`bin/`](bin) | The tools a host's units and releases run, listed below |

## Tools on a host

| Tool | What it does |
|---|---|
| `flatshare-release` | Prepares, verifies and switches a host's releases, and switches back on a failure |
| `serving-check` | Fails an engine's start when what it runs differs from the serving profile |
| `flatshare-tunnel` | Serves a GPU node's three sockets on the control host through SSH |
| `flatshare-gateway-reload`, `flatshare-gateway-stop` | The gateway unit's reload, a handover with no gap, and its stop after a drain |
| `gpu-guard` | Refuses to start a GPU server while another process holds the GPU |
| `in-private-net`, `private-net-check` | Run a GPU server in a network namespace of its own, and say whether the host allows one |
| `in-voicenet`, `voicenet-holder` | Hold the voice plane's network namespace, which has no route out, and run a command in it |
| `audit-listeners` | Fails unless the host's own processes listen only where they may |
| `wait-healthy` | Blocks a unit's start until its server answers |
| `node-report` | Prints what a soak reads of a GPU node at each check |
| `flatshare-svc`, `shims/` | Run the units on a host without a systemd user manager |
| `flatshare-forward`, `flatshare-local` | Relay between sockets and ports, and run a host's command locally, in single-host mode |
