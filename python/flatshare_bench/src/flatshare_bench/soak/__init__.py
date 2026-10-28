"""``flatshare-bench soak``: the production path under load for hours, every attempt and every scrape kept.

E2's points (:mod:`flatshare_bench.e2`) measure a fixed point for a few minutes on a box switched into its bench mode:
the engine's development routes, its prefix cache emptied, the gateway's canary off, the voice worker taking every
call. A soak measures the box as it serves, and changes nothing on it: on cyan, with the voice worker in serve mode
(its call-capacity admission) and the gateway and engines as deployed, it runs

- **calls** (:mod:`flatshare_voice.arrivals`, inside the voice-plane namespace): synthetic callers arriving as a
  Poisson process for the soak's whole length, each call of a few turns with the callers' think times, so that some
  find the worker full; every attempt is recorded, accepted, refused (and why) or failed;
- **agents**: AIPerf's agentic replay as E2 runs it, through the gateway as the agents tenant, started again when it
  ends before the soak does, each of its runs recorded;
- **scrapes** (:mod:`.scrape`): every minute the gateway's own counts and the engines' metrics, the voice worker's
  status, and the state and journal of cyan's flatshare units;
- **a guard** (:mod:`.guard`): the soak's load stops while another's process holds a GPU it loads, and that is an
  event of the record.

:mod:`.run` drives it, :mod:`.summary` reads its directory (or the compact record it exports) into a summary per hour
and overall. It takes cyan's lock, as an E2 point and a drill do, so that none of them overlaps a soak.
"""
