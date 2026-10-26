#!/usr/bin/env bash
# push.sh HOST: copy to a single host what its bootstrap and its campaign read from this workstation: this
# repository to HOST:~/flatshare/src, as deploy/deploy.sh syncs it, with the commit in src/DEPLOYED (which run
# records name); the corpora and the callers' audio (weka-x32, weka-x32-long and callers under FLATSHARE_DATA, default
# ~/flatshare/data) to HOST:~/flatshare/data; and FLATSHARE_GATEWAY_BIN, a flatshare-gateway build to try before
# bootstrap.sh builds one on the host. rsync's: a rerun copies only what changed. FLATSHARE_SSH is your ssh command for
# the host, as for deploy/deploy.sh. Then, on the host: bash flatshare/src/deploy/a100/bootstrap.sh stage
set -euo pipefail
host=${1:?usage: push.sh HOST}
repo=$(cd "$(dirname "$0")/../.." && pwd)
ssh=${FLATSHARE_SSH:-ssh}
data=${FLATSHARE_DATA:-$HOME/flatshare/data}
for name in weka-x32 weka-x32-long callers; do
  [[ -d $data/$name ]] ||
    { echo "push: no $data/$name (FLATSHARE_DATA names the directory that holds it)" >&2; exit 2; }
done
$ssh "$host" mkdir -p flatshare/src flatshare/data flatshare/bin
rsync -a --delete -e "$ssh" --exclude .git --exclude .claude --exclude target --exclude __pycache__ --exclude '*.egg-info' \
  --exclude .venv --exclude .pytest_cache --exclude .ruff_cache "$repo/" "$host:flatshare/src/"
rev=$(git -C "$repo" describe --always --dirty 2>/dev/null || echo unknown)
$ssh "$host" "echo $rev >flatshare/src/DEPLOYED"
# -L: as files, the links' targets. weka-x32-long is long_traces.py's links into this workstation's weka-x32 by
# absolute path, which would dangle on the host; a copy that an earlier push made of the links is replaced.
for name in weka-x32 weka-x32-long callers; do
  rsync -aL -e "$ssh" "$data/$name" "$host:flatshare/data/"
done
if [[ -n ${FLATSHARE_GATEWAY_BIN:-} ]]; then
  rsync -a -e "$ssh" "$FLATSHARE_GATEWAY_BIN" "$host:flatshare/bin/flatshare-gateway"
fi
echo "push: $host has the source at $rev and the data; next, on it: bash flatshare/src/deploy/a100/bootstrap.sh stage"
