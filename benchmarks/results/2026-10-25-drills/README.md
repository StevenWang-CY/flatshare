# Failure drill records (2026-10-25)

The records behind [BENCHMARKS](../../../docs/BENCHMARKS.md#failure-drills)' failure drills: the deployed box on the
configuration to deploy (make-room with the PRs' switches and a 48-token window), Phastform's RTX 5060 Ti behind cyan's
gateway, on four builds. [RUNBOOK.md](../../../ops/RUNBOOK.md#drills) says how to run each one. Home paths are written
`~`, and the `/dev/shm` listings keep only this project's files (the host's other users' are cut).

| Set | Build, times (UTC) | What it shows |
|---|---|---|
| `engine-kill/`, `gateway-restart/`, `cpu-tier/`, `start-engine/`, `start-engine-cold/` | W8's, 03:50–04:10Z, before W6's hardening | a killed engine broke every stream with no error event; a restart drained the gateway's streams while it refused new connections |
| `w6-*` | v0.3-191-g987fd09, 12:28–12:31Z: the servers on Unix sockets, `flatshare-gateway.socket` holding the gateway's listeners, a reload handing them to a successor | a reload failed no probe; a restart after it cut its streams (below) |
| `v0.3-274-*` | v0.3-274-gd6d6814, 17:12–17:16Z | a restart after a reload drained (the unit's `ExecStop`, `deploy/bin/flatshare-gateway-stop`, waits for the drain), and 5 of its 42 probes failed with 502: cyan's tunnel let a burst of new connections pass a master's sessions (below) |
| `v0.3-295-*` | v0.3-295-ga2db355, 19:27–19:31Z, the final build | the drills as BENCHMARKS gives them: every drill as `v0.3-274-*`'s, and the restart's probes all answered |

| Directory | Drill | Its files |
|---|---|---|
| `engine-kill/` | `ops/drills/drill.sh engine-kill`: SIGKILL to every process of `flatshare-engine` while two realtime and four background streams decode | `drill.json` (each stream and probe attempt, wall-clock), `report.txt` (`flatshare-bench drill report`), `event` (the kill's time), `engine-journal.txt` (the unit's journal from the kill on), `requests-*.txt` (the engine's running and waiting requests before, after and at the end), `shm-*.txt` (`/dev/shm`'s used and size bytes, then our files), `scheduler` (the engine's scheduler class) |
| `gateway-restart/` | `ops/drills/drill.sh gateway-restart`: `systemctl --user restart flatshare-gateway` on cyan, the same streams | as above, without the engine's journal |
| `cpu-tier/` | `flatshare-bench drill cpu-tier`: twice the CPU tier's capacity of unique prompts, then the latest restored and the first evicted | `tier.json` |
| `start-engine/`, `start-engine-cold/` | `ops/drills/cold-start.sh engine` and `engine --cold` (the compile and JIT caches emptied first, then put back) | `report.txt` (the phases), `lh-engine-journal.txt` |
| `w6-engine-kill/` | `engine-kill` on the final build | as `engine-kill/`, the engine's running and waiting requests sampled every 0.5 s in `drill.json` in place of `requests-*.txt` |
| `w6-gateway-reload/` | `ops/drills/drill.sh gateway-reload`: `systemctl --user reload flatshare-gateway`, as a deploy takes a new gateway | `drill.json`, `report.txt`, `event`, `event.log` (the reload's own output), `gateway-journal.txt` (from the event on), `client.log`, `shm-*.txt`, `scheduler` |
| `w6-gateway-restart/` | `gateway-restart`, 30 s after the reload | as `w6-gateway-reload/` |
| `w6-gateway-kill/` | `ops/drills/drill.sh gateway-kill`: SIGKILL to the gateway, which systemd starts again a second later while its socket unit holds the listeners | as `w6-gateway-reload/` |
| `w6-start-box/` | `ops/drills/cold-start.sh box`: `systemctl --user restart flatshare.target`, the STT and TTS, then the engine | `report.txt` (each unit's phases), each unit's journal |
| `v0.3-274-*/`, `v0.3-295-*/` | the same five drills on those builds; the box's start takes the three servers at once | as the `w6-*` directories of the same name |

What the records do not show:
- **W8's gateway restart's outage (`gateway-restart/`).** The drill's client runs its streams, then its probe, so the
  gateway-restart probe started once the drain had let every stream finish, and succeeded at once. Requests sent
  during the drain fail, and none was. `requests-after-2s.txt` was taken 2 s after `systemctl restart` returned, which
  was after the drain (about 25 s after the event). Since W6 the drills probe from the event on.
- **W8's box start.** W8's `cold-start.sh box` restarted the engine alone (the audio servers' stops were cancelled by
  the start that followed them; fixed in 6b6554d), so its record is left out; the `*-start-box/` directories are the
  box's start.
- **A drained restart on v0.3-191.** `w6-gateway-restart/` ran 30 s after the reload drill, and its streams were cut
  11–12 ms after the restart instead of drained: the reload's successor was not a process systemd had started, and
  systemd 249's stop does not wait for such a main process, so it sent SIGKILL right after SIGTERM, as
  `gateway-journal.txt` shows. From v0.3-274 on, the unit's `ExecStop` sends the SIGTERM and waits for the drain
  itself, and the restart after the reload drains (`v0.3-274-gateway-restart/`, `v0.3-295-gateway-restart/`).
- **The restart's 502s on v0.3-274.** The new gateway opened its upstream connections at once, about 40 through the
  engine's socket, and cyan's tunnel sent each one past a master's sessions to the node on its own, where sshd dropped
  those past its MaxStartups: 5 of the 42 probes, queued in the socket's backlog during the drain, got 502. The tunnel
  now waits for a master with room (2932cbe), and the final build's restart failed none.
- **The STT's graph capture on v0.3-191.** `w6-start-box/report.txt` puts the STT's "CUDA graphs are captured" at
  55.4 s, before its KV cache (55.8 s): the STT profiles its graphs' memory with a capture first, and the report then
  took the first capture line; its real capture ends at 57.4 s (`lh-stt-journal.txt`). The later builds' reports take
  the last.
