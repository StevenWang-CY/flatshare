"""conn-cost.py OUT: what a new connection through flatshare-tunnel costs. For each of the GPU node's sockets on this
host, 50 GET /health on one kept-alive connection, then 50 each on a connection of its own (Connection: close, so the
server ends it and the next request opens one: one SSH session through the tunnel). Each mode has a client of its own,
whose first request, unmeasured, opens the kept connection or ends the first new one, so every measured "new" request
opens a connection. p95 is the nearest rank (the 48th of 50). Prints and writes OUT (conn-cost.json)."""

import json
import math
import os
import statistics
import sys
import time

import httpx

N = 50
out = {}
for name in ("engine", "stt", "tts"):
    path = f"/run/user/{os.getuid()}/flatshare/phastform-{name}.sock"
    rows = {}
    for mode, headers in (("kept", {}), ("new", {"Connection": "close"})):
        transport = httpx.HTTPTransport(uds=path)
        with httpx.Client(transport=transport, base_url="http://localhost") as client:
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
