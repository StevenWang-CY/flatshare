"""Check that installed distributions are as their wheels shipped them: every file in each one's RECORD hashes to
the recorded sha256. A stock vLLM is one whose files all match.

usage: python check_record.py DIST...   (exit 1 if any file differs or is missing)
"""

import base64
import hashlib
import sys
from importlib.metadata import distribution


def check(name):
    dist = distribution(name)
    bad, checked = [], 0
    for f in dist.files or []:
        if not f.hash or f.hash.mode != "sha256":
            continue
        path = f.locate()
        try:
            digest = hashlib.sha256(path.read_bytes()).digest()
        except FileNotFoundError:
            bad.append(f"missing {f}")
            continue
        checked += 1
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != f.hash.value:
            bad.append(f"differs {f}")
    print(f"{name} {dist.version}: {checked} files checked, {len(bad)} differ or are missing")
    for b in bad[:20]:
        print("  ", b)
    return not bad


if __name__ == "__main__":
    results = [check(n) for n in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
