#!/usr/bin/env bash
# deploy.sh HOST: sync this repository's commit to HOST:~/leaseholder/src, install HOST's units (deploy/<short-name>/),
# and audit what listens on HOST (deploy/bin/audit-listeners). The sync carries the commit HEAD names and nothing else
# (git archive): no untracked or ignored file, no __pycache__, no change that is not committed. A tree with changes the
# sync would not carry is refused, so that DEPLOYED, the commit synced, names what HOST runs. Set LH_SSH to the ssh
# command for your hosts, e.g. LH_SSH="ssh -F $HOME/.ssh/penn_config".
set -euo pipefail
host=${1:?usage: deploy.sh HOST}
name=${host%%.*}
repo=$(cd "$(dirname "$0")/.." && pwd)
ssh=${LH_SSH:-ssh}
[[ -d $repo/deploy/$name ]] || { echo "deploy: no deploy/$name directory" >&2; exit 2; }
changes=$(git -C "$repo" status --porcelain --untracked-files=normal)
if [[ -n $changes ]]; then
  echo "deploy: the tree has changes the sync would not carry; commit or stash them:" >&2
  echo "$changes" >&2
  exit 2
fi
# The commit synced, which run records name (lh-bench e2 run).
rev=$(git -C "$repo" describe --always)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
git -C "$repo" archive HEAD | tar -x -C "$stage"
$ssh "$host" mkdir -p leaseholder/src
# --delete drops what the commit no longer has, the host's __pycache__ among it; a Rust build there stays.
rsync -a --delete --exclude /target -e "$ssh" "$stage/" "$host:leaseholder/src/"
$ssh "$host" "echo $rev >leaseholder/src/DEPLOYED"
$ssh "$host" "bash leaseholder/src/deploy/$name/install.sh"
# Units this install changed take effect when they restart: an audit that fails names what still listens.
$ssh "$host" "python3 leaseholder/src/deploy/bin/audit-listeners" ||
  { echo "deploy: $host: a process of the box's units listens where it may not (above)" >&2; exit 1; }
