# W8 failure drill records (2026-10-25)

The records behind [BENCHMARKS](../../../docs/BENCHMARKS.md#failure-drills-w8)' failure drills: the deployed box on the
configuration to deploy (make-room with the PRs' switches and a 48-token window), Phastform's RTX 5060 Ti behind cyan's
gateway, 03:50–04:10Z. [RUNBOOK.md](../../../ops/RUNBOOK.md#drills) says how to run each one. Home paths are written
`~`, and the `/dev/shm` listings keep only this project's files (the host's other users' are cut).

| Directory | Drill | Its files |
|---|---|---|
| `engine-kill/` | `ops/drills/drill.sh engine-kill`: SIGKILL to every process of `lh-engine` while two realtime and four background streams decode | `drill.json` (each stream and probe attempt, wall-clock), `report.txt` (`lh-bench drill report`), `event` (the kill's time), `engine-journal.txt` (the unit's journal from the kill on), `requests-*.txt` (the engine's running and waiting requests before, after and at the end), `shm-*.txt` (`/dev/shm`'s used and size bytes, then our files), `scheduler` (the engine's scheduler class) |
| `gateway-restart/` | `ops/drills/drill.sh gateway-restart`: `systemctl --user restart lh-gateway` on cyan, the same streams | as above, without the engine's journal |
| `cpu-tier/` | `lh-bench drill cpu-tier`: twice the CPU tier's capacity of unique prompts, then the latest restored and the first evicted | `tier.json` |
| `start-engine/`, `start-engine-cold/` | `ops/drills/cold-start.sh engine` and `engine --cold` (the compile and JIT caches emptied first, then put back) | `report.txt` (the phases), `lh-engine-journal.txt` |

What the records do not show:
- **A gateway restart's outage.** The drill's client runs its streams, then its probe, so the gateway-restart probe
  started once the drain had let every stream finish, and succeeded at once. Requests sent during the drain fail, and
  none was. `requests-after-2s.txt` was taken 2 s after `systemctl restart` returned, which was after the drain (about
  25 s after the event).
- **The box's start.** `cold-start.sh box` of 10-25 restarted the engine alone (the audio servers' stops were cancelled
  by the start that followed them; fixed in 6b6554d), so its record is left out until it runs again.
