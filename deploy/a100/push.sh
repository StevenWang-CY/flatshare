#!/usr/bin/env bash
# push.sh HOST: copy to a single host what its bootstrap and its campaign read from this workstation: this
# repository to HOST:~/leaseholder/src, as deploy/deploy.sh syncs it, with the commit in src/DEPLOYED (which run
# records name); the corpora and the callers' audio (weka-x32, weka-x32-long and callers under LH_DATA, default
# ~/leaseholder/data) to HOST:~/leaseholder/data; and LH_GATEWAY_BIN, an lh-gateway build to try before bootstrap.sh
# builds one on the host. rsync's: a rerun copies only what changed. LH_SSH is your ssh command for the host, as for
# deploy/deploy.sh. Then, on the host: bash leaseholder/src/deploy/a100/bootstrap.sh stage
set -euo pipefail
host=${1:?usage: push.sh HOST}
repo=$(cd "$(dirname "$0")/../.." && pwd)
ssh=${LH_SSH:-ssh}
data=${LH_DATA:-$HOME/leaseholder/data}
for name in weka-x32 weka-x32-long callers; do
  [[ -d $data/$name ]] || { echo "push: no $data/$name (LH_DATA names the directory that holds it)" >&2; exit 2; }
done
$ssh "$host" mkdir -p leaseholder/src leaseholder/data leaseholder/bin
rsync -a --delete -e "$ssh" --exclude .git --exclude .claude --exclude target --exclude __pycache__ --exclude '*.egg-info' \
  --exclude .venv --exclude .pytest_cache --exclude .ruff_cache "$repo/" "$host:leaseholder/src/"
rev=$(git -C "$repo" describe --always --dirty 2>/dev/null || echo unknown)
$ssh "$host" "echo $rev >leaseholder/src/DEPLOYED"
# -L: as files, the links' targets. weka-x32-long is long_traces.py's links into this workstation's weka-x32 by
# absolute path, which would dangle on the host; a copy that an earlier push made of the links is replaced.
for name in weka-x32 weka-x32-long callers; do
  rsync -aL -e "$ssh" "$data/$name" "$host:leaseholder/data/"
done
if [[ -n ${LH_GATEWAY_BIN:-} ]]; then
  rsync -a -e "$ssh" "$LH_GATEWAY_BIN" "$host:leaseholder/bin/lh-gateway"
fi
echo "push: $host has the source at $rev and the data; next, on it: bash leaseholder/src/deploy/a100/bootstrap.sh stage"
