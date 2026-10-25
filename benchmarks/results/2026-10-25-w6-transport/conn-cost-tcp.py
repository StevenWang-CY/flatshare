"""conn-cost-tcp.py OUT PORTS: what a new connection costs on the transport A/B's TCP side (W5's): the GPU node's
sockets through one SSH connection's local forwards, as W5's tunnel unit forwarded them, one channel per connection.
PORTS is engine=PORT,stt=PORT,tts=PORT on 127.0.0.1. For each, conn-cost.py's two modes: 50 GET /health on one
kept-alive connection, then 50 each on a connection of its own (Connection: close), each mode with a client of its
own whose first request, unmeasured, opens the kept connection or ends the first new one. p95 is the nearest rank
(the 48th of 50). Prints and writes OUT."""

import json
import math
import statistics
import sys
import time

import httpx

N = 50
ports = dict(item.split("=") for item in sys.argv[2].split(","))
out = {}
for name in ("engine", "stt", "tts"):
    rows = {}
    for mode, headers in (("kept", {}), ("new", {"Connection": "close"})):
        with httpx.Client(base_url=f"http://127.0.0.1:{ports[name]}") as client:
            client.get("/health", headers=headers).raise_for_status()
            times = []
            for _ in range(N):
                started = time.perf_counter()
                client.get("/health", headers=headers).raise_for_status()
                times.append((time.perf_counter() - started) * 1000)
                time.sleep(0.05)
        times.sort()
        rows[mode] = {
            "n": N,
            "p50": round(statistics.median(times), 2),
            "p95": round(times[math.ceil(0.95 * N) - 1], 2),
            "max": round(times[-1], 2),
        }
    out[name] = rows
    print(name, json.dumps(rows))
json.dump(out, open(sys.argv[1], "w"), indent=1)
