"""Write replies.tsv: every voice reply W5's gateway completed, with its output tokens, which the compact records do not
keep. It ran once, on the raw run directories of the v1.0 release's asset leaseholder-raw-records-w5.tar.gz (each
point's gateway.jsonl, the gateway's request log): python3 replies.py RUNS > replies.tsv, with RUNS the asset's
unpacked w5/ directory. A reply is a line of the voice tenant's /v1/chat/completions that completed, and its tokens
are the completion_tokens the gateway logged from the engine's usage; the warm-up turns' replies are among them, and
every one of them ended at a stop. The 87 requests the voice agent cancelled before they completed log no usage and
are left out."""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = ("campaign", "point", "completion_tokens")


def replies(runs: Path) -> list[tuple[str, str, int]]:
    """Each voice reply of each point under ``runs`` (CAMPAIGN/POINT/gateway.jsonl), in file order."""
    rows = []
    for log in sorted(runs.glob("*/*/gateway.jsonl")):
        campaign, point = log.parent.parent.name, log.parent.name
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            if (
                entry.get("tenant") == "voice"
                and entry.get("route") == "/v1/chat/completions"
                and entry.get("outcome") == "completed"
            ):
                rows.append((campaign, point, int(entry["completion_tokens"])))
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python3 replies.py RUNS > replies.tsv")
    print("\t".join(COLUMNS))
    for row in replies(Path(sys.argv[1])):
        print("\t".join(str(value) for value in row))


if __name__ == "__main__":
    main()
